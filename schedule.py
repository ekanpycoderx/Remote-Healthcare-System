from ast import For
import time
import sys
import os
import datetime
from turtle import st
from click import open_file
import colorama
from colorama import Fore,Style,Back
from matplotlib.font_manager import FontEntry
import csv
import start

def check_schedule():
    return None

def update_schedule():
    print(Fore.GREEN + 'INFO : Kindly select how you want to upload the schedule.')
    print(Fore.GREEN + 'Type \'man\' for manual updating or type \'auto\' for auto updateion : ')
    way = input('> ')
    if way == 'man':
        print(Fore.GREEN + 'INFO : Manual mode selected .. ')
    elif way == 'auto':
        print(Fore.GREEN + 'INFO : Auto mode selected')
        print('Select the file to upload the schedule : ')

def show_schedule():
    os.system('schedule.csv')
    os.system('cls')
    start.home_page()

def make_sample_schedule():
    print(Fore.GREEN + 'INFO : Producing a sample Schedule.. ')
    filename = 'schedule.csv'
    time.sleep(2)
    fields = ['Activity', 'Time Slot', 'Motionlessness', 'Duration']
    rows = [
	['Waking up','7 AM - 7:30 AM', 'True', '30m'],
	['Exercise','8 AM - 9:30 AM', 'False', '1.5h'],
	['Bathing','9:45 AM - 10:15 AM', 'False','30m'],
	['Breakfast','10:30 AM - 11 AM', 'False/Probably', '30m'],
	['Job','11 AM - 12:45 PM', 'False', '1.65h'],
	['Nap','12:45 PM - 1 PM', 'True', '15m'],
	['Lunch','1 PM - 2 PM', 'False/Probably', '1h'],
	['Nap','2 PM - 3 PM', 'True', '1h'],
	['Evening stuffs','3:30 PM - 7:30 PM', 'False', '4h'],
	['Dinner','8 PM - 9 PM', 'False', '1h'],
	['Sleep', '10 PM - 7 AM', 'True', '9h']]
    with open(filename, 'w', newline='') as csvfile:
        write = csv.writer(csvfile)
        write.writerow(fields)
        write.writerows(rows)
        print(Fore.GREEN + 'SUCESS : Schedule created sucessfully! Now you can edit the schedule or add more tasks')
        print(Fore.BLUE + 'INFO : The file name is stored as schedule.csv')
        open_file = input('Open the file to check/edit? : ')
        csvfile.close()
        if open_file == 'y':
            os.system('schedule.csv')
            os.system('cls')
            start.home_page()
        elif open_file == 'n':
            os.system('cls')
            start.home_page()


# SAMPLE EXAMPLE FOR MAKING SCHEDULES
# # field names
# fields = ['Name', 'Branch', 'Year', 'CGPA']
	
# # data rows of csv file
# rows = [ ['Nikhil', 'COE', '2', '9.0'],
# 		['Sanchit', 'COE', '2', '9.1'],
# 		['Aditya', 'IT', '2', '9.3'],
# 		['Sagar', 'SE', '1', '9.5'],
# 		['Prateek', 'MCE', '3', '7.8'],
# 		['Sahil', 'EP', '2', '9.1']]
	
# # name of csv file
# filename = "university_records.csv"
	
# # writing to csv file
# with open(filename, 'w') as csvfile:
# 	# creating a csv writer object
# 	csvwriter = csv.writer(csvfile)
		
# 	# writing the fields
# 	csvwriter.writerow(fields)
		
# 	# writing the data rows
# 	csvwriter.writerows(rows)