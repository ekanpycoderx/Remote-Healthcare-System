from enum import auto
import os
import cv2
import colorama
from colorama import Fore

colorama.init(autoreset=True)

def camera():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3,1280)
    camera_video.set(4,960)
    print(Fore.BLUE + 'INFO : Camera being set up ... ')
    while camera_video.isOpened():
        # Read a frame.
        ok, frame = camera_video.read()
    
        # Check if frame is not read properly.
        if not ok:
        
            # Continue to the next iteration to read the next frame and ignore the empty camera frame.
            continue
    
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        print(Fore.GREEN + 'INFO : Camera Detected and Working.')
        fps = str(camera_video.get(cv2.CAP_PROP_FPS))
        fps_text = f'FPS : {fps}'
        cv2.putText(frame, fps_text, (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0), 2)
        directions = 'Press Escape key to capture an image'
        cv2.putText(frame, directions, (20,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0), 2)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(5) & 0xFF == 27: #27 means esc key
            break
    cv2.imwrite('Client.jpg', frame)
    camera_video.release()
    cv2.destroyAllWindows()
    os.system('cls')
    print(Fore.GREEN + 'INFO : Photo named Client.jpg has been saved sucessfully')
    #home_page.home_page()

camera()