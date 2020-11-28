# steering.py
from flask import Flask, render_template, Response
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
import keyboard

app = Flask(__name__)
app.config["DEBUG"] = True


camera = cv2.Videocapture(0)
currentKey = list()

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            currentKey = list()
            key = False

            img = np.flip(img,axis=1)
            img = imutils.resize(img, width=640)
            img = imutils.resize(img, height=480)

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            value = (11, 11)
            blurred = cv2.GaussianBlur(hsv, value,0)
            colourLower = np.array([53, 55, 209])
            colourUpper = np.array([180,255,255])

            height = img.shape[0]
            width = img.shape[1]

            mask = cv2.inRange(blurred, colourLower, colourUpper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

            upContour = mask[0:height//2,0:width]
            downContour = mask[3*height//4:height,2*width//5:3*width//5]

            cnts_up = cv2.findContours(upContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts_up = imutils.grab_contours(cnts_up)


            cnts_down = cv2.findContours(downContour, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts_down = imutils.grab_contours(cnts_down)

            if len(cnts_up) > 0:
        
                c = max(cnts_up, key=cv2.contourArea)
                M = cv2.moments(c)
                cX = int(M["m10"]/(M["m00"]+0.000001))

                if cX < (width//2 - 35):
                    keyboard.press('a')
                    key = True
                    currentKey.append('a')
                elif cX > (width//2 + 35):
                    keyboard.press('d')
                    key = True
                    currentKey.append('d')
                    
            
            if len(cnts_down) > 0:
                keyboard.press('Space')
                key = True
                currentKey.append('Space')
            
            img = cv2.rectangle(img,(0,0),(width//2- 35,height//2 ),(0,255,0),1)
            cv2.putText(img,'LEFT',(110,30),cv2.FONT_HERSHEY_DUPLEX,1,(139,0,0))

            img = cv2.rectangle(img,(width//2 + 35,0),(width-2,height//2 ),(0,255,0),1)
            cv2.putText(img,'RIGHT',(440,30),cv2.FONT_HERSHEY_DUPLEX,1,(139,0,0))

            img = cv2.rectangle(img,(2*(width//5),3*(height//4)),(3*width//5,height),(0,255,0),1)
            cv2.putText(img,'NITRO',(2*(width//5) + 20,height-10),cv2.FONT_HERSHEY_DUPLEX,1,(139,0,0))

            if not key and len(currentKey) != 0:
                for current in currentKey:
                    ReleaseKey(current)
                currentKey = list()

            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('home.html')


if __name__ == '__main__':
    app.run(threaded=True, port=5000)