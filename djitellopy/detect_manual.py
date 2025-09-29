from djitellopy import Tello
from ultralytics import YOLO
import cv2, math, time
from datetime import datetime


tello = Tello()
tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()


model = YOLO("yolov8n.pt")  

tello.takeoff()
while True:
    
    img = frame_read.frame
    if img is None:
        print("No frame received")
        continue

    results = model.predict(source=img, conf=0.5, show=False, stream=True)

    for result in results:
        boxes = result.boxes.cpu().numpy()  
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0]  
            cls = int(box.cls[0]) 
           
            if cls == 0:
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Person: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("drone", img)

    key = cv2.waitKey(1) & 0xff
    if key == 27:  
        break
    elif key == ord('w'):
        tello.move_forward(60)
    elif key == ord('s'):
        tello.move_back(60)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
        # for _ in range(12):  # 12 steps of 30 degrees each
        #     tello.rotate_clockwise(30)
        #     time.sleep(0.5)  # Add a short delay between rotations
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.move_up(30)
    elif key == ord('f'):
        tello.move_down(30)
    elif key == ord('l'): 
        tello.land()
        print("Drone landed")


tello.land()
cv2.destroyAllWindows()
