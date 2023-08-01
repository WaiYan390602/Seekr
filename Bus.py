# Interpreter provides methods to interact with ML Model
from tflite_runtime.interpreter import Interpreter

# array operations library
import numpy as np

# image processing library
import cv2

# time measurement library
import time

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

				if Object[9] == 1 : # and Object[0] == maxObj[0] :
					# calculate IoU of maxObj and Object
					curIoU = IoU(maxObj, Object)

					# if IoU > threshold, remove current object from the list
					if curIoU > IoUthreshold :
						objects[index2][9] = 0
						num -= 1

		index += 1

	return finalObjs



# get model paths and image path from the user

# initialize argument parser object
parser = argparse.ArgumentParser("Detects bus panels and extracts bus numbers")

# specify arguments to be parsed
parser.add_argument('panel_model', type = str, help="file path of the ML model for detecting bus panels")
parser.add_argument('number_model', type = str, help="file path of the ML model for extracting bus numbers")
parser.add_argument('image', type = str, help="file path of the image")

# parse arguments 
args = parser.parse_args()

# file paths to the ML models
panel_model = args.panel_model
number_model = args.number_model
# file path to the image
img_path = args.image

""" # camera object for accessing the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened() :
	print("Cannot open camera")
	exit() """

# Interpreter objects
panel_interpreter = Interpreter(model_path = panel_model)
number_interpreter = Interpreter(model_path = number_model)
# list of alphanumerics recognized by number_model
ANlabels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B' , 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
	    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Thresholds
PconfidenceThd = 60
NconfidenceThd = 60
PscoreThd = 30
NscoreThd = 50
IoUthreshold = 0.5

PscalingCoeff = 640/109 #224/166
NscalingCoeff = 640/240
# print(scalingCoeff)

# printing the structure of panel_model's input
Pinput_tensor = panel_interpreter.get_input_details()
print("\n(Input Details of Panel Model)")
print(Pinput_tensor, '\n')

# extracting the shape of panel_model's input
Pinput_shape = Pinput_tensor[0]['shape']
print(Pinput_shape, '\n')
PimH = Pinput_shape[1]
PimW = Pinput_shape[2]

# printing the structure of panel_model's output
Poutput_tensor = panel_interpreter.get_output_details()
print("(Output Details of Panel Model)")
print(Poutput_tensor, '\n')



# printing the structure of number_model's input
Ninput_tensor = number_interpreter.get_input_details()
print("\n(Input Details of Number Model)")
print(Ninput_tensor, '\n')

# extracting the shape of number_model's input
Ninput_shape = Ninput_tensor[0]['shape']
print(Ninput_shape, '\n')
NimH = Ninput_shape[1]
NimW = Ninput_shape[2]

# printing the structure of number_model's output
Noutput_tensor = number_interpreter.get_output_details()
print("(Output Details of Number Model)")
print(Noutput_tensor, '\n')



# allocate tensors
panel_interpreter.allocate_tensors()
number_interpreter.allocate_tensors()

# loading and reformatting an image to be fed as the input to panel_model
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (PimW, PimH))
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

	# img = cv2.resize(frame, (PimW, PimH))
	# imgF = np.array(img)
	# now the image is stored as an array in imgF

	# add an extra dimension to img array (** what is batch dimension?)
	# processed_img = np.expand_dims(imgF, axis=0)

	# run the panel_model

	# set the input tensor
	panel_interpreter.set_tensor(Pinput_tensor[0]['index'], processed_img)

	# run inference and calculate time elapsed
	t1 = time.time()
	panel_interpreter.invoke()
	t2 = time.time()

	time_elapsed = (t2-t1) * 1000 # in milliseconds

	print("Time taken for Inference (panel_model) = ", time_elapsed, " ms\n")



	t3 = time.time()
	results = panel_interpreter.get_tensor(Poutput_tensor[0]['index'])[0]
	rows = Poutput_tensor[0]['shape'][1]
	columns = Poutput_tensor[0]['shape'][2]
	# print(rows, ' ', columns)
	
	# filter detected panels based on confidence and score
	panels = filter(results, rows, columns, PimW, PimH, PscalingCoeff, PconfidenceThd, PscoreThd)

	# sort the panels based on confidence
	def takeConf(panel) :
		return panel[1]

	panels.sort(reverse=True, key=takeConf)

	# for panel in panels :
		# print(panel)
	# print('\n')

	# apply non maximum suppression to already sorted list of bus panels
	finalPanels = NMS(panels, IoUthreshold)
	
	# sort final panels from right to left
	def PtakeXC(panel) :
		return panel[2]
		
	finalPanels.sort(reverse=True, key=PtakeXC)

	# for panel in finalPanels :
		# print(panel)
	# print('\n')



	# draw bounding boxes for detected panels and crop out their images

	# list for cropped bus panels
	panels = []
	# list of numbers detected in each panel
	# numList format -> [count of alphanumerics, corresponding ANs, pairs in same format]
	numList = []

	# width and height of the image to be displayed
	# disW = 500
	# disH = 500
	xrscF = 1 # disW/PimW
	yrscF = 1 # disH/PimH

	# img = cv2.resize(img, (disW, disH))

	for panel in finalPanels :
		# Get bounding box coordinates and draw box
		x1 = int(panel[4]*xrscF + 0.5)
		y1 = int(panel[5]*yrscF + 0.5)

		x2 = int(panel[6]*xrscF + 0.5)
		y2 = int(panel[7]*yrscF + 0.5)

		# cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		croppedImg = img[y1 : y2, x1 : x2]
		panels.append(croppedImg)

	t4 = time.time()

	print("Time taken for post processing (panel_model) = ", (t4-t3) * 1000, " ms\n")



	# now resize cropped images of bus panels and give them as inputs to number_model

	# width and height of the image to be displayed
	disW = NimW
	disH = NimH
	xrscF = disW/NimW
	yrscF = disH/NimH

	index = 1
	for panel in panels :
		panel = cv2.resize(panel, (NimW, NimH))
		tmp = np.array(panel)
		tmp = np.expand_dims(tmp, axis=0)

		number_interpreter.set_tensor(Ninput_tensor[0]['index'], tmp)

		t5 = time.time()
		number_interpreter.invoke()
		t6 = time.time()

		time_elapsed = (t6-t5) * 1000 # in milliseconds

		print("Time taken for Inference (number_model) = ", time_elapsed, " ms\n")

		t7 = time.time()
		results = number_interpreter.get_tensor(Noutput_tensor[0]['index'])[0]
		rows = Noutput_tensor[0]['shape'][1]
		columns = Noutput_tensor[0]['shape'][2]
		# print(rows, ' ', columns)

		print("Processing panel(" + str(index) + ")\n")

		# filter detected alphanumerics based on confidence and score
		alphanumerics = filter(results, rows, columns, NimW, NimH, NscalingCoeff, NconfidenceThd, NscoreThd)

		# sort the alphanumerics based on confidence
		def takeConf(alphanumeric) :
			return alphanumeric[1]

		alphanumerics.sort(reverse=True, key=takeConf)

		# for alphanumeric in alphanumerics :
			# print(alphanumeric)
		# print('\n')

		# apply non maximum suppression to already sorted list of alphanumerics
		finalANs = NMS(alphanumerics, IoUthreshold)
		
		# sort alphanumerics from left to right
		def ANtakeXC(AN) :
			return AN[2]
		
		finalANs.sort(reverse=False, key=ANtakeXC)

		# draw bounding boxes for detected alphanumerics

		panel = cv2.resize(panel, (disW, disH))

		numList.append(len(finalANs))
		for AN in finalANs :
			numList.append(AN[0])
			# print(AN)
			# Get bounding box coordinates and draw box
			x1 = int(AN[4]*xrscF + 0.5)
			y1 = int(AN[5]*yrscF + 0.5)

			x2 = int(AN[6]*xrscF + 0.5)
			y2 = int(AN[7]*yrscF + 0.5)

			cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
		
		t8 = time.time()

		print("Time taken for post processing (number_model) = ", (t8-t7) * 1000, " ms\n")

		cv2.imshow("Panel-" + str(index), panel)

		index += 1

	# cv2.imshow("Detected Panels", img)

	# print('\n', numList, '\n')

	print("The buses are : ")
	i = 0
	while i < len(numList) :

		length = numList[i]
		i += 1
		
		for j in range(i, i+length) :
			print(ANlabels[numList[j]], end="")
		print('\n')

		i += length

	# break

	if cv2.waitKey(0) :
	# if cv2.waitKey(1) == ord('q') :
		break
	# cv2.destroyAllWindows()

# When everything done, release the camera object and close all windows
# camera.release()
cv2.destroyAllWindows()
