from djitellopy import Tello
from ultralytics import YOLO
import cv2, time, csv, os
from datetime import datetime

tello = Tello()
tello.connect()

print("Battery:", tello.get_battery())

tello.streamon()
frame_read = tello.get_frame_read()

model = YOLO("yolov8n.pt")  

if not os.path.exists("logs"):
    os.makedirs("logs")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/detections_{timestamp}.csv"

log_file = open(log_filename, mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "class", "confidence", "x", "y", "w", "h", "action",
                     "height", "temp", "speed_x", "speed_y", "speed_z", "flight_time"])

tello.takeoff()

while True:
    img = frame_read.frame
    if img is None:
        print("No frame received")
        continue

    height = tello.get_height()
    temp = tello.get_temperature()
    speed_x = tello.get_speed_x()
    speed_y = tello.get_speed_y()
    speed_z = tello.get_speed_z()
    flight_time = tello.get_flight_time()

    # if battery < 15:
    #     print("⚠️ Low battery, landing!")
    #     tello.land()
    #     csv_writer.writerow([
    #         time.time(), "none", 0, -1, -1, -1, -1, "emergency_land",
    #         battery, height, temp, speed_x, speed_y, speed_z, flight_time
    #     ])
    #     break


    results = model.predict(source=img, conf=0.5, show=False, stream=True)

    action = "none" 

    for result in results:
        boxes = result.boxes.cpu().numpy()  
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = float(box.conf[0])  
            cls = int(box.cls[0])  

            w = x2 - x1
            h = y2 - y1
            x = x1 + w // 2
            y = y1 + h // 2

            if cls == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Person: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                frame_center_x = img.shape[1] // 2
                offset = x - frame_center_x

                if offset > 100:  # person is to the right
                    tello.move_right(20)
                    action = "move_right"
                elif offset < -100:  # person is to the left
                    tello.move_left(20)
                    action = "move_left"
                else:
                    tello.move_forward(20)
                    action = "move_forward"

                # Log detection + action
                csv_writer.writerow([time.time(), cls, conf, x, y, w, h, action,
                     height, temp, speed_x, speed_y, speed_z, flight_time                
                                     ])
                log_file.flush()

    # if not detected:
    #     tello.send_rc_control(0, 0, 0, 0)  # hover
    #     csv_writer.writerow([
    #         time.time(), "none", 0, -1, -1, -1, -1, "hover",
    #         battery, height, temp, speed_x, speed_y, speed_z, flight_time
    #     ])
    #     log_file.flush()

    cv2.imshow("Drone YOLO Feed", img)

    key = cv2.waitKey(1) & 0xff
    if key == 27:  # ESC
        break
    elif key == ord('l'): 
        tello.land()
        action = "manual_land"
        csv_writer.writerow([time.time(), "manual", 1.0, -1, -1, -1, -1, action,
                             height, temp, speed_x, speed_y, speed_z, flight_time])
        log_file.flush()

tello.land()
cv2.destroyAllWindows()
log_file.close()
