from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2


encode_path = './image_list_encoding.pickle'
display = 1

print("Loading encodings")
data = pickle.loads(open(encode_path, "rb").read())

print("Starting video stream")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

while True:
	frame = vs.read()
	
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	face_location_boxes = face_recognition.face_locations(rgb, model="cnn")
	encoded_faces = face_recognition.face_encodings(rgb, face_location_boxes)
	names = []

	for encoded_face in encoded_faces:
		matches, face_dist_to_conf = face_recognition.compare_faces(data["encodings"], encoded_face)
		name = "Unknown person"

		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			name = max(counts, key=counts.get)
		
		names.append(name)

	for ((top, right, bottom, left), name) in zip(face_location_boxes, names):
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		if "/" in name:
			accuracy = int(max(face_dist_to_conf)*100)
			name = "Name:{} Accuracy:{}%".format(name.split("/")[2], str(accuracy))
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	if display > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

cv2.destroyAllWindows()
vs.stop()

if writer is not None:
	writer.release()