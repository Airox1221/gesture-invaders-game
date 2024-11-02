import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the path to the CSV file
csv_file_path = 'finger_nodes.csv'

# Check if the CSV file already exists. If not, create it and add headers.
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header for CSV file
        writer.writerow(['x5', 'y5', 'x8', 'y8', 'x9', 'y9', 'x12', 'y12',
                         'x13', 'y13', 'x16', 'y16', 'x17', 'y17', 'x20', 'y20', 'class'])

# Set up video capture
cap = cv2.VideoCapture(0)

# Set up MediaPipe Hands model
with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    i = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the image color space from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hand landmarks
        results = hands.process(rgb_frame)

        # Check if landmarks are detected

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Collect coordinates of the specified nodes
                selected_nodes = []
                for idx in [5, 8, 9, 12, 13, 16, 17, 20]:  # Indices for the knuckles and fingertip nodes
                    h, w, _ = frame.shape
                    x, y = int(hand_landmarks.landmark[idx].x * w), int(hand_landmarks.landmark[idx].y * h)
                    selected_nodes.extend([x, y])

                # Append the class label '0'
                selected_nodes.append(0)

                # Write data to CSV
                with open(csv_file_path, mode='a', newline='') as file:
                    if i % 10 == 0:
                        writer = csv.writer(file)
                        writer.writerow(selected_nodes)

                print("Written to CSV:", selected_nodes)
                i += 1

        # Show the frame with landmarks
        cv2.imshow('Finger Nodes', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
