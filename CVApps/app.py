#from cv.faceDetect import faceDetection
from flask import Response
from flask import Flask
from flask import render_template
import threading
import imutils
from imutils.video import VideoStream
import datetime
import time
import cv2


outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detectFace():
	global vs,outputFrame,lock

	face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
	total=0
	while True:
		frame=vs.read()
		frame=imutils.resize(frame,width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces=face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(80, 80),flags=0)
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			roi_gray=gray[y:y+h,x:x+w]
			roi_color=frame[y:y+h,x:x+w]
			eyes=eye_cascade.detectMultiScale(roi_gray)
			for(ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		with lock:
			outputFrame=frame.copy()

def generate():
	global outputFrame,lock

	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			if not flag:
				continue

			yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
	app.run(debug=True)