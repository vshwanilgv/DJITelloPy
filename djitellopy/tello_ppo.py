import cv2
import numpy as np
from djitellopy import Tello
from stable_baselines3 import PPO
import torch
from ultralytics import YOLO
import time
import os

# Ensure display environment on macOS
os.environ['DISPLAY'] = ':0'  # For XQuartz, if needed

# Initialize Tello
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")
tello.streamon()

# Initialize YOLO (update path to your trained model)
yolo_model = YOLO("yolov8n.pt")  # Replace with your model path

# Initialize PPO model
ppo_model = PPO.load("gymnasium/ppo_human_nav")

# Camera parameters
frame_width = 960
frame_height = 720
frame_center_x = frame_width / 2
frame_center_y = frame_height / 2

# Safety parameters
max_height = 100  # 1m in cm
min_distance_bbox_size = 0.7
max_steps = 30
move_distance = 20  # cm

def get_yolo_state(frame):
    """Process frame with YOLO, return state [dx, dy, bbox_size] for largest person or None."""
    results = yolo_model(frame)
    max_area = 0
    best_state = None
    for result in results:
        boxes = result.boxes.xywh.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            if result.names[int(cls)] == "person":
                x, y, w, h = box
                area = w * h
                if area > max_area:
                    max_area = area
                    dx = (x - frame_center_x) / (frame_width / 2)
                    dy = (y - frame_center_y) / (frame_height / 2)
                    bbox_size = np.clip((w * h) / (frame_width * frame_height) * 8, 0, 1)  # Adjusted multiplier
                    best_state = np.array([np.clip(dx, -1, 1), np.clip(dy, -1, 1), bbox_size], dtype=np.float32)
    return best_state

def draw_yolo_boxes(frame, results, state):
    """Draw bounding boxes and state info on frame."""
    frame_copy = frame.copy()
    for result in results:
        boxes = result.boxes.xywh.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        for box, cls in zip(boxes, classes):
            if result.names[int(cls)] == "person":
                x, y, w, h = box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_copy, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if state is not None:
        state_text = f"dx: {state[0]:.2f}, dy: {state[1]:.2f}, bbox_size: {state[2]:.2f}"
        cv2.putText(frame_copy, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame_copy

def map_action_to_tello(action):
    """Map PPO action to Tello command."""
    if action == 0:  # Forward
        return lambda: tello.move_forward(move_distance)
    elif action == 1:  # Back
        return lambda: tello.move_back(move_distance)
    elif action == 2:  # Left
        return lambda: tello.move_left(move_distance)
    elif action == 3:  # Right
        return lambda: tello.move_right(move_distance)
    elif action == 4:  # Hover
        return lambda: tello.send_rc_control(0, 0, 0, 0)

# Initialize OpenCV window
cv2.namedWindow("Tello Feed", cv2.WINDOW_NORMAL)

# Open log file
with open("tello_log.txt", "w") as log_file:
    try:
        # Takeoff
        tello.takeoff()
        print("Takeoff complete, stabilizing...")
        log_file.write("Takeoff complete, stabilizing...\n")
        time.sleep(3)  # Extended stabilization

        tello.set_speed(10)
        steps = 0
        while steps < max_steps:
            # Get frame
            frame = tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Get state
            state = get_yolo_state(frame)
            results = yolo_model(frame)  # For visualization
            frame_with_boxes = draw_yolo_boxes(frame, results, state)
            cv2.imshow("Tello Feed", frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if state is None:
                print("No human detected, hovering...")
                log_file.write(f"Step {steps}: No human detected, hovering...\n")
                tello.send_rc_control(0, 0, 0, 0)
                steps += 1
                time.sleep(0.5)
                continue

            print(f"Step {steps}, State {state}")
            log_file.write(f"Step {steps}, State {state}\n")

            # Safety: Hover if too close
            if state[2] > min_distance_bbox_size:
                print("Too close, hovering...")
                log_file.write(f"Step {steps}: Too close, hovering...\n")
                tello.send_rc_control(0, 0, 0, 0)
                steps += 1
                time.sleep(0.5)
                continue

            # Check height
            current_height = tello.get_height()
            if current_height > max_height:
                print(f"Height {current_height}cm, adjusting down...")
                log_file.write(f"Step {steps}: Height {current_height}cm, adjusting down...\n")
                tello.move_down(20)
                time.sleep(1)
                continue
            elif current_height < 30:
                print(f"Height {current_height}cm, adjusting up...")
                log_file.write(f"Step {steps}: Height {current_height}cm, adjusting up...\n")
                tello.move_up(20)
                time.sleep(1)
                continue

            # Get action from PPO
            action, _ = ppo_model.predict(state, deterministic=True)
            print(f"Step {steps}, Action {action}, Height {current_height}cm")
            log_file.write(f"Step {steps}, Action {action}, Height {current_height}cm\n")

            # Execute action with retry
            try:
                tello_action = map_action_to_tello(action)
                tello_action()
            except Exception as e:
                print(f"Action failed: {e}, retrying as hover...")
                log_file.write(f"Step {steps}: Action failed: {e}, retrying as hover...\n")
                tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)

            steps += 1

    except Exception as e:
        print(f"Error: {e}")
        log_file.write(f"Error: {e}\n")
    finally:
        tello.send_rc_control(0, 0, 0, 0)
        for _ in range(3):  # Retry landing
            try:
                tello.land()
                break
            except Exception as e:
                print(f"Land failed: {e}, retrying...")
                log_file.write(f"Land failed: {e}, retrying...\n")
                time.sleep(1)
        try:
            tello.streamoff()
        except Exception as e:
            print(f"Streamoff failed: {e}")
            log_file.write(f"Streamoff failed: {e}\n")
        cv2.destroyAllWindows()