import cv2
import face_recognition
import face_recognition as fr
import numpy as np
from colorama import Fore, Back, Style
import time
import os
import shutil
import mediapipe as mp 
import start
import camera
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./trained"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names

def classify_face_vid(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = im
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)
    # while True:
    #   cv2.imshow('Video', img)
    #   if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

def face_detector_img():
  image = input('Enter the sample image to be detected : ')
  print(classify_face(image))

def face_trainer_file():
  print(Fore.GREEN + 'INFO : Select the image to be trained .. ')
  print(Fore.BLUE + ''' NOTE : Kindly note the following details while uploading the face to be trained : 
  1. The image should be cropped and only one face should be shown at a time to avoid background noises 
  which may lead to inefficiency and inaccuracy of the AI.
  2. Remove any sort of filter to avoid confusion and pain for the AI.''')
  Tk().withdraw()
  filename = askopenfilename()
  print(f'The file selected is : {filename}')
  confirm = input('Confirm the file? (type \'y\' or \'n\') > ')
  if confirm == 'y':
    print(Fore.GREEN + 'INFO : Image confirmed!')
    name = input('Enter Client\'s name for training : ')
    target = r'C:\Users\ekans\Desktop\RHS\trained\%s.jpg' %(name)
    shutil.copyfile(filename, target)
    print(Fore.GREEN + 'INFO : Training in progress .. ')
    time.sleep(1)
    print(Fore.GREEN + 'INFO : Image trained sucessfully!')
  elif confirm == 'n':
    print(Fore.RED + 'ERROR : Selection failed.. ')
    time.sleep(2)
    os.system('cls')
    face_trainer_file()
  else:
    print(Fore.RED + 'ERROR : INVALID STATEMENT')
    face_trainer_file()

def face_train_camera():
  return None

def video_face_detector_new():
    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():
        # Read a frame.
        ok, frame = camera_video.read()
    
        # Check if frame is not read properly.
        if not ok:
        
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
    
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
    
        # # Get the width and height of the frame
        # frame_height, frame_width, _ =  frame.shape
    
        # # Resize the frame while keeping the aspect ratio.
        # frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        face_frame = classify_face_vid(frame)
    
        # Display the frame.
        cv2.imshow('Face Classification', face_frame)
        # fps = camera_video.get(cv2.CAP_PROP_FPS)
        # cv2.putText(frame, fps, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        # k = cv2.waitKey(1) & 0xFF
    
        # # Check if 'ESC' is pressed.
        # if(k == 27):
        #   break
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Release the VideoCapture object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()
    start.home_page()


def trial_face_detect_img():
  img_dir = input('Enter directory for the image to test: ')
  IMAGE_FILES = [img_dir]
  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils
  with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
      image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
      if not results.detections:
        continue
      annotated_image = image.copy()
      for detection in results.detections:
        print('Nose tip: ')
        print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
        mp_drawing.draw_detection(annotated_image, detection)
      cv2.imwrite(str(idx) + '.png', annotated_image)
      print(Fore.GREEN + 'INFO : Annotations has been done sucessfully')
      start.home_page()
