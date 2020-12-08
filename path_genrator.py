import os             
# all_files = os.listdir("labels/") 

path="C:\\Users\\Saurav Akolia\\Google Drive\\License_Plate\\Dataset\\validation\\Vehicle registration plate"
os.chdir(path)
# all_files=[]
# import os
# for file in os.listdir("img"):
#     if file.endswith(".txt"):
#        all_files.append(file)
	# replacement strings
# WINDOWS_LINE_ENDING = b'\r\n'
# UNIX_LINE_ENDING = b'\n'


import glob
import os

# os.chdir(r'directory where the files are located')
myFiles = glob.glob('*.jpg')
# print(myFiles[0][-5:-4])

write_file="valid.txt"
content="data/test/"
content_valid="/content/drive/MyDrive/License_Plate/Dataset/validation/Vehicle_registration_plate/"
	
with open(write_file, 'w+') as open_file:

	for x in myFiles:
		if x==write_file:
			continue
		else:
			print(content+x)
			open_file.write(content+x)
			open_file.write("\n")
	

# for deleting file
# for x in myFiles:
# 	if x[-5:-4]==')':
# 		print(x)
# 		os.remove(os.path.join(path,x))