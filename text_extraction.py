import cv2
import numpy as np 
import argparse
import time
import os
import re
import matplotlib.pyplot as plt

import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def load_image(image_path):
	original_image = cv2.imread(image_path)
	# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	# image_data = cv2.resize(original_image, (416, 416))
	# image_data = image_data / 255.
	return original_image


def text_getter(image):
	# gray = cv2.imread("C:\\Users\\Saurav Akolia\\Google Drive\\License_Plate\\croped.jpg", 0)
	gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
	# cv2.imshow("black",gray)
	# cv2.imshow("gray",gray)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	gray = cv2.medianBlur(gray, 3)
	
	# perform otsu thresh (using binary inverse since opencv contours work better with white text)
	ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
	# print("thresh")
	# cv2.imshow("thresh",thresh)
	# cv2.waitKey(0)
	rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

	# apply dilation 
	dilation = cv2.dilate(thresh, rect_kern, iterations = 2)
	cv2.imshow("dilation",dilation)
	cv2.imshow("gray",gray)
	#cv2.waitKey(0)
	# find contours
	try:
	  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	except:
	  ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

	# create copy of image
	im2 = gray.copy()
	pre_text=pytesseract.image_to_string(im2, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
	pre_text = re.sub('[\\W_]+', '', pre_text)
	# print("K")
	plate_num = ""
	
	# loop through contours and find letters in license plate
	for cnt in sorted_contours:

		# print("sd")
		x,y,w,h=cv2.boundingRect(cnt)
		height, width = im2.shape
		# if height of box is not a quarter of total height then skip
		if height / float(h) > 6: continue
	  	
		ratio=h / float(w)

	  	# if height to width ratio is less than 1.5 skip

		if ratio <1.5:continue

		area=h * w

	# if width is not more than 25 pixels skip
		if width / float(w) > 15: continue
	  	
	# if area is less than 100 pixels skip
		if  area < 100: continue
		# draw the rectangle

		rect=cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)

		roi=  thresh[y-5:y+h+5, x-5:x+w+5]

		roi=cv2.bitwise_not(roi)

		roi=cv2.medianBlur(roi, 5)

		text=pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')

		clean_text = re.sub('[\\W_]+', '', text)
	 	
		plate_num += clean_text

	if len(plate_num)==0:
		return pre_text
	else:
		return plate_num


 

def plate_dtection():

	CONFIDENCE_THRESHOLD = 0.2
	NMS_THRESHOLD = 0.4
	COLORS = [(0, 255, 255)]

	class_names = []

	# import unidecode

	classes = []
	net = cv2.dnn.readNet("C:\\Users\\Saurav Akolia\\Google Drive\\License_Plate\\backup\\yolov4-obj_final.weights", "C:\\Users\\Saurav Akolia\\Google Drive\\License_Plate\\yolov4-obj.cfg")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(size=(416, 416), scale=1/255)

	image=load_image("us.jpg")

	classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

	with open("C:\\Users\\Saurav Akolia\\Google Drive\\License_Plate\\obj.names", "r") as f:
		# class_name=f.readlines()
			class_names = [line.strip() for line in f.readlines()]

	for (classid, score, box) in zip(classes, scores, boxes):
	  color = COLORS[int(classid) % len(COLORS)]
	  
	  # print(label)
	  xmin, ymin, xmax, ymax = box
	  ymin=int(ymin)
	  xmax=int(xmax)
	  ymin=int(ymin)
	  ymax=int(ymax)

	  x,y,w,h=box
	  label = "%s : %s"  % (class_names[classid[0]],score)

	  cropped_img=image[y:y+h, x:x+w]
	 
	   # save image
	  plate_no=text_getter(cropped_img)
	  

	  print(plate_no)
	  cv2.imwrite("cropped_images\\croped.jpg", cropped_img)
	  cv2.rectangle(image, (xmin,ymin), (xmin+xmax, ymin+ymax), (0,0,255), 2)

	  
	  cv2.putText(image,label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
	  cv2.putText(image,plate_no, (box[0]+10, box[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,120),2 )
	
	cv2.imwrite("Detected_images\\det.jpg",image)  
	cv2.imshow("image",image)
	cv2.waitKey(0)



plate_dtection()