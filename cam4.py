import torch
import cv2
import pandas as pd
import csv
from datetime import datetime
import time
import os

# Load the YOLOv8 model from the ultralytics library
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Change to yolov5m, yolov5l, yolov5x, or custom model

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' for a video file

# Initialize variables for counting persons and for time tracking
person_count = 0
start_time = time.time()

# Initialize list to store data
data = []

# Function to save data to CSV
def save_to_csv(data):
    file_exists = os.path.isfile('person_count_log.csv')
    with open('person_count_log.csv', 'a', newline='') as csvfile:
        fieldnames = ['Time', 'Person Count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(data)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame using the YOLOv8 model
    results = model(frame)

    # Reset person count for each frame
    person_count = 0

    # Loop over the detections
    for result in results.xyxy[0]:  # For each detection
        class_id = int(result[5])
        confidence = result[4].item()
        if confidence > 0.2 and class_id == 0:  # Confidence threshold and class ID for person
            person_count += 1
            x1, y1, x2, y2 = map(int, result[:4])

            # Draw the bounding box around the detected person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the person count on the frame
    cv2.putText(frame, f'Total Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Save person count every 3 seconds
    current_time = time.time()
    if current_time - start_time >= 3:
        start_time = current_time
        new_entry = {'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Person Count': person_count}
        data.append(new_entry)
        save_to_csv(data)
        data = []  # Reset data after saving

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
