
import csv
	
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
	['Sleep', '10 PM - 7 AM', 'True', '9h']
]

