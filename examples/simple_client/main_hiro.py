import dataclasses
import enum
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

#realsense imports
import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import json

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    model_host: str = "0.0.0.0"
    # Port to connect to the policy server.
    model_port: int | None = 8000
    franka_host: str = "0.0.0.0"
    # Port to connect to the policy server.
    franka_port: int | None = 8000
    # Number of steps to run the policy for.
    num_steps: int = 100
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None

def get_joint_positions(host, port):
    url = f"http://{host}:{port}/joint_states"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract joint positions from the response
            joint_positions = [
                data["q"][i] for i in range(7)
            ]
            joint_positions = np.array(joint_positions)
            return joint_positions
            
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except KeyError as e:
        print(f"Missing key in response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def set_joint_velocities(host, port, velocities):

    # Define the URL and headers
    url = f"http://{host}:{port}/joint_velocities"
    headers = {
        "Content-Type": "application/json"
    }

    # Define the payload
    payload_velocity = []
    for v in velocities:
        payload_velocity.append({"velocities" : v})
    payload = {
        "velocity_commands": payload_velocity,
        "duration_per_command": 0.1,
        "ramp_down_time": 0.001
    }

    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)
        
        # Print response details
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("Motion completed successfully!")
        else:
            print(f"Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:


    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.model_host,
        port=args.model_port,
        api_key=None
    )
    logger.info(f"Server metadata: {policy.get_server_metadata()}")

    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device('943222071556')
    config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # ...from Camera 2
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device('838212073252')
    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    # Start streaming from both cameras
    pipeline_1.start(config_1)
    pipeline_2.start(config_2)

    obs_fn = lambda: get_observations(args, pipeline_1, pipeline_2)


    # Send a few observations to make sure the model is loaded.
    for _ in range(2):
        policy.infer(obs_fn())

    timing_recorder = TimingRecorder()

    for _ in tqdm.trange(args.num_steps, desc="Running policy"):
        action = policy.infer(obs_fn())
        print(action['actions'].shape)
        print(action['actions'][0,:])
        actions_array = action['actions']
        actions_array= actions_array / 5.0
        actions_array = actions_array[:, 0:7].tolist()
        set_joint_velocities(args.franka_host,args.franka_port, actions_array)

    # for _ in tqdm.trange(args.num_steps, desc="Running policy"):
    #     inference_start = time.time()
    #     action = policy.infer(obs_fn())
    #     timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
    #     for key, value in action.get("server_timing", {}).items():
    #         timing_recorder.record(f"server_{key}", value)
    #     for key, value in action.get("policy_timing", {}).items():
    #         timing_recorder.record(f"policy_{key}", value)

    # timing_recorder.print_all_stats()

    # if args.timing_file is not None:
    #     timing_recorder.write_parquet(args.timing_file)


def get_observations(args, pipeline_1, pipeline_2):

    frames_1 = pipeline_1.wait_for_frames()
    frames_2 = pipeline_2.wait_for_frames()
    
    color_frame_1 = frames_1.get_color_frame()
    color_frame_2 = frames_2.get_color_frame()
    joint_pos = get_joint_positions(args.franka_host, args.franka_port)
    if joint_pos is None:
        raise Exception("Could not get joint positions")
    
    if  not color_frame_1 or not color_frame_2:
        raise Exception("no color frame")

    color_image_1 = np.asanyarray(color_frame_1.get_data())
    color_image_2 = np.asanyarray(color_frame_2.get_data())


    resized_1 = cv2.resize(color_image_1, (224, 224))
    resized_2 = cv2.resize(color_image_2, (224, 224))

    return {
        "observation/exterior_image_1_left" : resized_1,
        "observation/wrist_image_left": resized_2,
        "observation/joint_position": joint_pos,
        "observation/gripper_position": np.random.rand(1),
        "prompt": "touch the cup",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
