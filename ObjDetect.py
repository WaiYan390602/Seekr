# Interpreter provides methods to interact with ML Model
from tflite_runtime.interpreter import Interpreter

# array operations library
import numpy as np

# image processing library
import cv2

# time measurement library
import time

# mathematics library
import math

# arguments processing library
import argparse



# functions

# function to filter output results based on confidence and score
def filter(results, rows, columns, imW, imH, scalingCoeff, confidenceThd, scoreThd) :

	# list to store filtered objects
	objects = []

	# go through each row of the output tensor
	for row in range(rows) :

		Id = 0
		maxScore = 0
		curConfidence = results[row][4]

		# if both confidence and score exceed thresholds, add object to filtered list
		if curConfidence >= confidenceThd :

			# finding the id of the object of current row
			for column in range(5, columns) :
				curScore = results[row][column]

				if curScore > maxScore :
					Id = column - 5
					maxScore = curScore

			# id and score of current row's object is stored in Id and maxScore respectively

			if maxScore > scoreThd :
				# bounding box coordinates
				# convert centre, width, height to two end points

				xc = results[row][0] * scalingCoeff
				yc = results[row][1] * scalingCoeff
				width = results[row][2] * scalingCoeff
				height = results[row][3] * scalingCoeff

				# upper left point
				x1 = max(0.0, xc - width/2)
				y1 = max(0.0, yc - height/2)

				# lower right point
				x2 = min(float(imW-1), xc + width/2)
				y2 = min(float(imH-1), yc + height/2)

				# format of an object -> [id, confidence, xcentre, ycentre, [coordinates of two corner points], area, status, score]
				objects.append([Id, curConfidence, xc, yc, x1, y1, x2, y2, (x2-x1) * (y2-y1), 1, maxScore])

	return objects

# function for calculating the IoU of two objects
def IoU(obj1, obj2) :
	# coordinates of intersecting bbox
	xx1 = max(obj1[4], obj2[4])
	yy1 = max(obj1[5], obj2[5])
	xx2 = min(obj1[6], obj2[6])
	yy2 = min(obj1[7], obj2[7])

	# find height and width of the intersection box
	# take max with 0.0 to avoid negative w and h due to non-overlapping boxes
	w = max(0.0, xx2 - xx1)
	h = max(0.0, yy2 - yy1)

	# intersection area
	IA = w * h

	UA = obj1[8] + obj2[8] - IA

	return IA/UA

# function to filter out overlapping bounding boxes
# non maximum suppression based on IoU (intersection over union) value
def NMS(objects, IoUthreshold) :

	# list to store final objects after NMS
	finalObjs = []

	num = len(objects)
	numcpy = num

	index = 0
	while num > 0 :

		# move object with highest confidence to final list
		# if it is still in the list
		maxObj = objects[index]
		if  maxObj[9] == 1 :
			objects[index][9] = 0
			num -= 1
			finalObjs.append(maxObj)


			for index2 in range(index+1, numcpy) :
				Object = objects[index2]

				if Object[9] == 1 and Object[0] == maxObj[0] :
					# calculate IoU of maxObj and Object
					curIoU = IoU(maxObj, Object)

					# if IoU > threshold, remove current object from the list
					if curIoU > IoUthreshold :
						objects[index2][9] = 0
						num -= 1

		index += 1

	return finalObjs

# function to calculate the distance to an object
coefficient = 139
def distance(objID, imgHeight) :
	return (coefficient * objs[objID][1]) / imgHeight

# function to calculate the angle (in degrees) where an object is located
# angle is calculated using the centre point of image's lower boundary as origin
def angle(xc, yc, model_width, model_height) :
    
	# threshold for object's x coordinate 
	threshold = 1e-3
  
	# x and y coordinates of the object (its centre point) relative to the origin above
	dx = xc - (model_width/2)
	dy = model_height - yc

	# if the x coordinate of the object is too close to 0, angle is 90 degrees
	if (dx < threshold) and (dx > -threshold) :
		return 90

	radAng = math.atan(dy/dx)

	# radAng < 0 means the object is in second quadrant
	# adjust angle value
	if radAng < 0 :
		radAng = math.pi + radAng

	# return angle value in degrees
	return radAng * (180 / math.pi)

# function to convert degrees to clock hand positions
# (0 to 180) -> (three to nine o'clock) in ccw direction
def degAng2clk(degree) :
	if degree >= 0 and degree < 15 :
		return 'three'
	elif degree >= 15 and degree < 45 :
		return 'two'
	elif degree >= 45 and degree < 75 :
		return 'one'
	elif degree >= 75 and degree < 105 :
		return 'twelve'
	elif degree >= 105 and degree < 135 :
		return 'eleven'
	elif degree >= 135 and degree < 165 :
		return 'ten'
	else :
		return 'nine'



# get model path and image path from the user

# initialize argument parser object
parser = argparse.ArgumentParser("Detects and gives directions to objects in an image")

# specify arguments to be parsed
parser.add_argument('model', type = str, help="file path of the ML model")
parser.add_argument('image', type = str, help="file path of the image")

# parse arguments 
args = parser.parse_args()

# file path to the ML model
model_path = args.model
# file path to the image
img_path = args.image

""" # camera object for accessing the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened() :
	print("Cannot open camera")
	exit() """

# Interpreter object
interpreter = Interpreter(model_path=model_path)
# list of labels and corresponding real heights of objects recognized by the model
objs = [("person", 1.7), ("bicycle", 1.676), ("car", 1.828), ("motorcycle", 1.524), ("airplane", 84.8), ("bus", 3.81), ("train", 4.025), ("truck", 1.940), ("boat", 0.46) , ("traffic light", 0.762),
          ("fire hydrant", 0.762), ("stop sign", 3.05), ("parking meter", 0.5), ("bench", 0.508), ("bird",  0.47), ("cat", 0.46), ("dog", 0.84), ("horse", 1.73), ("sheep", 1.17), ("cow", 1.8),
          ("elephant", 4), ("bear", 1.37), ("zebra", 1.91), ("giraffe", 6), ("backpack", 0.47), ("umbrella", 1.01), ("handbag", 0.23), ("tie", 0.508), ("suitcase", 0.56), ("frisbee", 0.3),
          ("skis", 1.58), ("snowboard", 1.35), ("sports ball", 0.23), ("kite", 0.9), ("baseball bat", 0.02), ("baseball glove", 0.3), ("skateboard", 0.0889), ("surfboard", 2.194),
          ("tennis racket", 0.6096), ("bottle", 0.304), ("wine glass", 0.155), ("cup", 0.094), ("fork", 0.18), ("knife", 0.152), ("spoon", 0.16), ("bowl", 0.1), ("banana", 0.18), ("apple", 0.02),
          ("sandwich", 0.121), ("orange", 0.02), ("broccoli", 0.121), ("carrot", 0.18), ("hot dog", 0.18), ("pizza", 0.02), ("donut", 0.0762), ("cake", 0.12), ("chair", 1.009), ("couch", 0.84),
          ("potted plant", 1.05), ("bed", 0.75), ("dining table", 0.787), ("toilet", 0.762), ("tv", 0.340), ("laptop", 0.209), ("mouse", 0.0381), ("remote", 0.17), ("keyboard", 0.022), 
          ("cell phone", 0.1436), ("microwave", 0.313), ("oven", 0.72), ("toaster", 0.284), ("sink", 0.75), ("refrigerator", 1.82), ("book", 0.215), ("clock", 0.3), ("vase", 0.4), 
          ("scissors", 0.1778), ("teddy bear", 0.304), ("hair drier", 0.28), ("toothbrush", 0.166)]
""" # printing labels and heights
for obj in objects :
	print('[ ', obj[0], ' ', obj[1], ' ]')
print('\n') """

# Thresholds
confidenceThd = 60
scoreThd = 60
IoUthreshold = 0.6

scalingCoeff = 224/167
# print(scalingCoeff)

# printing the structure of model's input
input_tensor = interpreter.get_input_details()
print("\n(Input Details of Model)")
print(input_tensor, '\n')

# extracting the shape of model's input
input_shape = input_tensor[0]['shape']
# print(input_shape, '\n')
imH = input_shape[1]
imW = input_shape[2]

# printing the structure of the model's output
output_tensor = interpreter.get_output_details()
print("(Output Details of Model)")
print(output_tensor, '\n')



# allocate tensors
interpreter.allocate_tensors()

# loading and reformatting an image to be fed as the input
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (imW, imH))
imgF = np.array(img)
# now the image is stored as an array in imgF

# add an extra dimension to img array (** what is batch dimension?)
processed_img = np.expand_dims(imgF, axis=0)

while(True) :
	# getting an image from camera and reformatting it to be fed as input
	# ret, frame = camera.read()

	# if frame is read correctly ret is True
	# if not ret:
		# print("Can't receive frame (stream end?). Exiting ...")
		# break

	# img = cv2.resize(frame, (imW, imH))
	# imgF = np.array(img)
	# now the image is stored as an array in imgF

	# add an extra dimension to img array (** what is batch dimension?)
	# processed_img = np.expand_dims(imgF, axis=0)



	# run the ML model

	# set the input tensor
	interpreter.set_tensor(input_tensor[0]['index'], processed_img)

	# run inference and calculate time elapsed
	t1 = time.time()
	interpreter.invoke()
	t2 = time.time()

	time_elapsed = (t2-t1) * 1000 # in milliseconds

	print("time taken for Inference = ", time_elapsed, " ms\n")



	t3 = time.time()
	results = interpreter.get_tensor(output_tensor[0]['index'])[0]
	rows = output_tensor[0]['shape'][1]
	columns = output_tensor[0]['shape'][2]
	# print(rows, ' ', columns)

	# filter the results based on confidence and score
	objects = filter(results, rows, columns, imW, imH, scalingCoeff, confidenceThd, scoreThd)

	# sort the objects based on confidence
	def takeConf(obj) :
		return obj[1]

	objects.sort(reverse=True, key=takeConf)

	for obj in objects :
		print(obj)
	print('\n')

	# apply non maximum suppression to already sorted list of objects
	finalObjs = NMS(objects, IoUthreshold)

	for Obj in finalObjs :
		print(Obj)
	print('\n')



	# draw bounding boxes for detected objects

	# width and height of the image to be displayed
	disW = 500
	disH = 500
	xrscF = disW/imW
	yrscF = disH/imH

	img = cv2.resize(img, (disW, disH))

	for Obj in finalObjs :
		# Give directions
		ang = angle(Obj[2], Obj[3], imW, imH)
		print("There is a ", objs[Obj[0]][0], " at ",  degAng2clk(ang), " o'clock (", ang, "degrees) and ", distance(Obj[0], Obj[7] - Obj[5]), " metres away from you\n")
		# Get bounding box coordinates and draw box
		x1 = int(Obj[4]*xrscF + 0.5)
		y1 = int(Obj[5]*yrscF + 0.5)
		
		x2 = int(Obj[6]*xrscF + 0.5)
		y2 = int(Obj[7]*yrscF + 0.5)
		
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		# Get centre point coordinates and draw angle line
		cv2.line(img, (int(disW/2), disH), (int(xrscF*Obj[2] + 0.5), int(yrscF*Obj[3] + 0.5)), (255, 0, 0), 2)

	t4 = time.time()

	print("Time taken for post processing = ", (t4-t3) * 1000, " ms\n")

	cv2.imshow('Detected Objects', img)

	if cv2.waitKey(0) :
	# if cv2.waitKey(1) == ord('q') :
		break
	# cv2.destroyAllWindows()

# When everything done, release the camera object and close all windows
# camera.release()
cv2.destroyAllWindows()
