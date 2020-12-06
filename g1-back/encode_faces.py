from imutils import paths
import face_recognition
import pickle
import cv2
import os

img_path = './image_list/'
encode_path = './image_list_encoding.pickle'

print("Getting faces")
imagePaths = list(paths.list_images(img_path))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	print("Image {}/{}".format(i + 1, len(imagePaths)))
	print(imagePath)
	name = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# resize image if it is big (CUDA Malloc)
	rows, cols, chan = rgb.shape
	if (rows * cols) > 1000000:
		scale_percent = 60 
		if (rows * cols) > 3000000:
			scale_percent = 30
		width = int(rgb.shape[1] * scale_percent / 100)
		height = int(rgb.shape[0] * scale_percent / 100)
		dim = (width, height)
		rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)


	boxes = face_recognition.face_locations(rgb, model="cnn")
	encodings = face_recognition.face_encodings(rgb, boxes)

	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}
f = open(encode_path, "wb")
f.write(pickle.dumps(data))
f.close()
