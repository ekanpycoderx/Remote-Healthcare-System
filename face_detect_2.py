import colorama
from colorama import Fore, Back, Style
import cv2
import start
import mediapipe as mp
import os

colorama.init(autoreset=True)

mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

# TRIAL PROGRAM TO SHOW FACE DETECTION
def trial_face_detect_video():
  cap = cv2.VideoCapture(0)
  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils
  with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print(Fore.RED + "INFO : Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image, detection)
          confidence = face_detection.predict(detection)
          print(f'Human has been detected with a confidence of {int(confidence)}')
      cv2.imshow('Video Face Detection', cv2.flip(image, 1))
      cv2.putText(image, 'Human Face', (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
      if cv2.waitKey(5) & 0xFF == ord('d'):
        break

  cap.release()
  cv2.destroyAllWindows()
  os.system('cls')
  start.home_page()