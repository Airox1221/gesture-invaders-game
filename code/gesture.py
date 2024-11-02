import cv2
import mediapipe as mp
import time
from decision import DecisionTreeModel

class HandTracker:

    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 10)
        if not self.cap.isOpened():
            print("Error: Camera not opened")
            return

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hand = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5,
                                        model_complexity=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.Dtm = DecisionTreeModel(data_file='finger_nodes.csv')
        self.Dtm.train()
        self.Dtm.load_model()

        # Time variables for FPS calculation
        self.c_time = 0
        self.pre_time = 0
        self.delay_time = 21  # Time delay between frames for smoother display
        # Finger position and image dimensions
        self.pos = [0, 0]  # Store finger position (X, Y)
        self.img_dim = []
        self.active_flag = False

    def process_frame(self, img):
        # Flip the image horizontally for a natural selfie view
        img = cv2.resize(img, (144, 144))
        img = cv2.flip(img, 1)

        # Check if the image is captured correctly
        if img is None:
            return img, [0, 0]

        try:
            # Convert BGR image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"Error converting to RGB: {e}")
            return img, [0, 0]

        # Process the image and detect hand landmarks
        result = self.hand.process(img_rgb)
        px, py = 0, 0
        sample_list = []

        # If hand landmarks are detected
        if result.multi_hand_landmarks:
            for frames in result.multi_hand_landmarks:
                for id, lm in enumerate(frames.landmark):
                    h, w, c = img.shape
                    self.img_dim = [h, w]
                    px, py = int(lm.x * w), int(lm.y * h)

                    # Track the index finger tip (landmark id 9)
                    if id == 9:
                        self.pos = [px, py]  # Update finger position

                    # Collect coordinates for specific landmarks: 5, 8, 9, 12, 13, 16, 17, 20
                    if id in [5, 8, 9, 12, 13, 16, 17, 20]:
                        sample_list.append(px)  # x-coordinate
                        sample_list.append(py)  # y-coordinate

                # Predict attack mode based on sample coordinates
                if sample_list:
                    attack_mode = self.Dtm.predict(sample_list)
                    print(f'class is {attack_mode}')
                    self.active_flag = (attack_mode == 0)
                else:
                    self.active_flag = False
        else:
            # Reset position if no hand is detected
            self.pos = [0, 0]
            self.active_flag = False

        cv2.waitKey(self.delay_time)

        # Calculate FPS
        # self.c_time = time.time()
        # fps = 1 / (self.c_time - self.pre_time)
        # self.pre_time = self.c_time
        # Display FPS on the image
        # cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        return img, self.pos

    def cleanup(self):
        # Release the camera and close any OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()
