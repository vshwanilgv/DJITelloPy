import cv2
from djitellopy import Tello
from ultralytics import YOLO
import numpy as np


tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()

model = YOLO("yolov8n.pt")  

try:
    while True:

        frame = frame_read.frame
        if frame is None:
            print("No frame received")
            continue

        frame_resized = cv2.resize(frame, (640, 480))

        # Perform YOLOv8 inference
        results = model.predict(source=frame_resized, conf=0.5, show=False, stream=True)

        # Process detections
        for result in results:
            boxes = result.boxes.cpu().numpy() 
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class ID

                # Filter for "person" class (class ID 0 in COCO dataset)
                if cls == 0:
                    # Draw bounding box and label
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_resized, f"Person: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the video feed with detections
        cv2.imshow("Tello YOLOv8 Human Detection", frame_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Release resources
    tello.streamoff()
    cv2.destroyAllWindows()