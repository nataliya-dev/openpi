import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()


ctx = rs.context()
devices = ctx.query_devices()
print(devices[0]) # 943222071556
print(devices[1]) # 838212073252

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



# exit()

# # Get device product line for setting a supporting resolution
# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
# pipeline_profile = config.resolve(pipeline_wrapper)
# device = pipeline_profile.get_device()

# # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames_1 = pipeline_1.wait_for_frames()
        frames_2 = pipeline_2.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        color_frame_2 = frames_2.get_color_frame()
        
        if  not color_frame_1 or not color_frame_2:
            print("error")
            continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        print(f"image size {color_image_1.shape}")

        color_image_2 = np.asanyarray(color_frame_2.get_data())
        print(f"image size {color_image_2.shape}")


        resized_1 = cv2.resize(color_image_1, (224, 224))
        print(f"resized size {resized_1.shape}")

        resized_2 = cv2.resize(color_image_2, (224, 224))
        print(f"resized size {resized_2.shape}")

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((resized_1, resized_2))


        # Show images
        cv2.imshow('RealSense', images)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
