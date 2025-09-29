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
csv_writer.writerow([
    "timestamp", "class", "confidence", "x", "y", "w", "h", "action",
    "height", "temp", "speed_x", "speed_y", "speed_z", "flight_time", "battery"
])

out = None  

tello.takeoff()

while True:
    img = frame_read.frame
    if img is None:
        print("No frame received")
        continue

    if out is None and img is not None:
        out = cv2.VideoWriter(
            f"logs/session_{timestamp}.avi",
            cv2.VideoWriter_fourcc(*'XVID'),
            30,  # FPS
            (img.shape[1], img.shape[0])  # width, height
        )

    height = tello.get_height()
    temp = tello.get_temperature()
    speed_x = tello.get_speed_x()
    speed_y = tello.get_speed_y()
    speed_z = tello.get_speed_z()
    flight_time = tello.get_flight_time()
    battery = tello.get_battery()

    # Low battery fail-safe
    # if battery < 15:
    #     print("⚠️ Low battery, landing!")
    #     tello.land()
    #     csv_writer.writerow([
    #         time.time(), "none", 0, -1, -1, -1, -1, "emergency_land",
    #         height, temp, speed_x, speed_y, speed_z, flight_time, battery
    #     ])
    #     break

    results = model.predict(source=img, conf=0.5, show=False, stream=True)

    action = "none"
    detected = False

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

            if cls == 0:  # person
                detected = True
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
                else:  # person is centered
                    tello.move_forward(20)
                    action = "move_forward"

                # Log detection
                csv_writer.writerow([
                    time.time(), cls, conf, x, y, w, h, action,
                    height, temp, speed_x, speed_y, speed_z, flight_time, battery
                ])
                log_file.flush()

    # If no detection → hover + log
    # if not detected:
    #     tello.send_rc_control(0, 0, 0, 0)
    #     action = "hover"
    #     csv_writer.writerow([
    #         time.time(), "none", 0, -1, -1, -1, -1, action,
    #         height, temp, speed_x, speed_y, speed_z, flight_time, battery
    #     ])
    #     log_file.flush()

    # Write video
    if out is not None:
        out.write(img)

    # Show video
    cv2.imshow("Drone YOLO Feed", img)

    key = cv2.waitKey(1) & 0xff
    if key == 27:  # ESC
        break
    elif key == ord('l'):
        tello.land()
        action = "manual_land"
        csv_writer.writerow([
            time.time(), "manual", 1.0, -1, -1, -1, -1, action,
            height, temp, speed_x, speed_y, speed_z, flight_time, battery
        ])
        log_file.flush()
        break

tello.land()
if out is not None:
    out.release()
cv2.destroyAllWindows()
log_file.close()
