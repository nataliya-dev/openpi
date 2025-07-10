import cv2

cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# Verify actual settings
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)


print(f"Actual resolution: {actual_width}x{actual_height} @ {actual_fps}fps")

while True:
    ret, frame = cap.read()

    print(frame)
    
    if ret:
        # Display the frame (optional)
        cv2.imshow('Logitech Camera', frame)
        
        # Press 's' to save current frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite('saved_frame.jpg', frame)
            print("Frame saved!")
        elif key == ord('q'):
            break
    else:
        print("Failed to read frame")
        break

cap.release()
cv2.destroyAllWindows()
