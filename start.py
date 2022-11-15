import colorama
from colorama import Fore, Back, Style
import time
from datetime import datetime
import sys
import os
import face_detect
import pose_detect
import face_detect_2
import schedule

colorama.init(autoreset=True)

#JUST FOR STYLING PURPOSES
st = 0.03
st1 = 0.01
def sp(str):
  for letter in str:
    sys.stdout.write(letter)
    sys.stdout.flush()
    time.sleep(st)
  print()
def sp1(str):
  for letter in str:
    sys.stdout.write(letter)
    sys.stdout.flush()
    time.sleep(st1)
  print()

# WELCOME SCRIPT
def welcomer():
  os.system('cls')
  print(Fore.BLUE + "                                       Welcome to Remote Healthcare System (RHS)                                       ")
  print('''                             Welcome to this newly made project on Remote Healthcare System               
                            This project has been made to serve the Senior Citizens and take             
                            care of their health without the requirement of any physical                 
                            supervisor to moniter their health.  The Program moniters and
                            schedules their activity and takes care of them. An Emergency 
                            response system has also been designed to react to situations
                            of emergency and call the nearest helpline.            \n''')
  time.sleep(2)
  home_page()

#TO BE OMITTED 

# # # LOGIN PAGE
# def login():
#   user = open("user.txt", "a+")
#   username = input("Username : ")
#   password = input("Password (press r to register if your are new) : ")
#   if username == "r":
#     register_user()
#     user.close()
#   elif username == user.readlines(0) and password == user.readlines(1):
#     print(Fore.GREEN + "Login sucessfull!")
#     sp1(f"Welcome {username}")
#     user.close()
#   else:
#     print("Invalid input.. RETRY")
#     print(f"username is and password is ")
#     user.read()
#     login()
#     user.close()

# # REGISTERING NEW USERS
# def register_user():
#   user = open("user.txt", "a+")
#   image_dir = open("dir.txt", "a+")
#   os.system("cls")
#   sp1("Hello! Kindly enter the follow details below to  continue registering.. ")
#   time.sleep(1)
#   new_username = input("Enter your new username (Note that it should be mathcing your original name for easier identification): ")
#   user.write(f"{new_username} \n")
#   new_pass = input("Enter a password for your account : ")
#   user.write(f"{new_pass} \n")
#   user.close()
#   sp1("Ok so now we are gonna require your images with only you and a simple background wiht solid color like green screen,etc.")
#   sp1("""Note that there should be any attachments used such as sunglasses,cap,etc this may
#   lead to bad results and the program may even fail completly""")
#   register_face()

# # REGISTERING NEW FACES
# def register_face():
#   retype_new_username = input("Enter username again to verify : ")
#   image_dir = open("dir.txt", "a+")
#   enter_dir = input("Enter directory for your image (Eg: G:\Ekanfiles\OpenCV-2\Model images) : ")
#   image_dir.write(f"{retype_new_username} -- {enter_dir} \n")
#   if retype_new_username == "exit":
#     os.system("exit")
#   elif NotADirectoryError(retype_new_username):
#     print(Fore.RED + "File not found!")
#     register_face()
#   else:
#     print("Congrats the face has been registered sucessfully!")
#     home_page()

# HOME PAGE
def home_page():
  os.system('cls')
  time.sleep(0.1)
  sp1(""" Select what you want to do -->""")
  print(Fore.GREEN + """ 
  [*] Live Video Cam (live_vid)
  
  [*] Check logs (chk_log)
  
  [*] Test the program (test_vid)

  [*] Test Video face detection (face_vid_new) (dlib used) (v2.[NEW!]

  [*] Face detection in images (face_img) [OLD]

  [*] Face detection in images (face_img_new) (dlib used) (v2.[NEW!]

  [*] Train AI for new Images (train_new) [NEW1]

  [*] Return to Home Page (welc)

  [*] Show / Check Schedules (chk_sch)

  [*] Update schedule/ Replace Schedule (up_sch)

  [*] Sample Schedule (samp_sch)
  
  [*] Exit (exit) \n""")
  print('Test for Developers --> ')
  print(Fore.BLUE + '''
  [*] Test/Sample for Pose detection for Video (pose_vid)

  [*] Test/Sample for pose detection for image (pose_img)

  [*] Test Video face detection (face_vid) \n''')
  print(Fore.BLUE + f'Current time is : {datetime.now()} \n')
  selected = input("Enter the function mentioned in the bracket to continue : ")
  while True:
    # if selected == "live_vid" :
    #   VideoCapture_Verifier()
    if selected == "chk_log":
      try:
        log = open("log.txt",'r+')
        for i in log.readlines():
          print(i)
      except FileNotFoundError:
        os.system('cls')
        print(Fore.CYAN + '''ERROR : A Log file does not exist the file might have been deleted 
or it does not exist this may happen if it is the first start for this program''')
      time.sleep(5)
      home_page()
    elif selected == "test_vid":
      face_detect_2.trial_face_detect_video()
    elif selected == 'chk_sch':
      schedule.show_schedule()
    elif selected == 'up_sch':
      schedule.update_schedule()
    elif selected == 'samp_sch':
      schedule.make_sample_schedule()
    elif selected == 'live_vid':
      pose_detect.live_vid_and_pose()
    elif selected == 'pose_vid':
      pose_detect.pose_detect_video()
    elif selected ==  'pose_img':
      pose_detect.pose_asker()
    elif selected == 'face_img':
      face_detect.trial_face_detect_img()
    elif selected == 'face_vid_new':
      face_detect.video_face_detector_new()
    elif selected == 'face_img_new':
      face_detect.face_detector_img()
    elif selected == 'face_vid':
      face_detect_2.trial_face_detect_video()
    elif selected == 'train_new':
      face_detect.face_trainer()
    elif selected == 'exit':
      os.system('cls')
      print(Fore.CYAN + 'INFO : Exiting the program.. ')
      sys.exit(0)
    elif selected == 'welc':
      welcomer()
    else:
      print(Fore.RED + "INVALID STATEMENT")
      os.system('cls')
      home_page()

def tester():
  os.system('cls')
  print("TESTING")
  for letter in ("##################### \n"):
      sys.stdout.write(letter)
      sys.stdout.flush()
      time.sleep(0.1)
  print(Fore.GREEN + "No error found :)")
  print(Fore.GREEN + "Test Sucessfull")
  time.sleep(2)
  os.system("cls")
  welcomer()


welcomer()
