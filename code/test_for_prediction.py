import mediapipe as mp
import cv2
from decision import DecisionTreeModel

dt_model = DecisionTreeModel(data_file='finger_nodes.csv')
dt_model.train()
dt_model.load_model()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # List to store specific x and y coordinates in a flat structure
    list_lm = []

    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Access only specific landmarks: 9, 10, and 20
            for index in [5, 8, 9, 12, 13, 16, 17, 20]:
                lm = hand_landmarks.landmark[index]

                # Convert normalized coordinates to pixel coordinates
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])

                # Append x and y directly to list_lm without any grouping
                list_lm.append(x)
                list_lm.append(y)

                # Draw circles on each selected landmark for visualization
                cv2.circle(frame, (x, y), 5, (0, 255, 0), cv2.FILLED)


    # Display the frame
    cv2.imshow("Hand Detection", frame)

    # Print the list to verify it's in the desired format
    #print("Landmark coordinates:", list_lm)  # [x9, y9, x10, y10, x20, y20]
    if len(list_lm) != 0:
                cls_decision = dt_model.predict(list_lm)
                print(cls_decision[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
