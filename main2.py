from flask import Flask, render_template,  Response, request
import cv2
import datetime, time
import os, sys
from threading import Thread
import numpy as np


app = Flask(__name__)

camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

global capture,rec_frame, grey, switch, neg, face, rec, out
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
sharp=0
Blur=0
Invert=0
sepia=0
sketch=0

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = generate_frames(frame)
                # frame = eye_detect(frame)
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (sepia):
                img_sepia = np.array(frame, dtype=np.float64)  # converting to float to prevent loss
                img_sepia = cv2.transform(img_sepia, np.matrix([[0.272, 0.534, 0.131],
                                                                [0.349, 0.686, 0.168],
                                                                [0.393, 0.769,
                                                                 0.189]]))  # multipying image with special sepia matrix
                img_sepia[np.where(img_sepia > 255)] = 255  # normalizing values greater than 255 to 255
                img_sepia = np.array(img_sepia, dtype=np.uint8)
                frame=img_sepia
            if (neg):
                frame = cv2.bitwise_not(frame)
                # frame = cv2.stylization(frame, sigma_s=60, sigma_r=0.07)
            if (Invert):
                frame = cv2.flip(frame, -1)
            if (sketch):
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9) # applying adaptive threshold to use it as a mask
                # color = cv2.bilateralFilter(frame, 9, 200, 200)
                # frame = cv2.bitwise_and(color, color, mask=edges) # cartoonize

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.medianBlur(gray, 7)
                frame = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
            if (Blur):
                frame = cv2.medianBlur(frame, 5)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray_blur = cv2.medianBlur(gray, 7)
                # frame = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
            if (capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['static', "shot_{}.png".format(str(now).replace(":", ''))])
                print(p)
                cv2.imwrite(p, frame)
            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                frame = cv2.flip(frame, 1)
            if (sharp):
                sharp_kernel = np.array([[0, -1, 0],
                                         [-1, 5, -1],
                                         [0, -1, 0]])
                # Sharpeneded image is obtained using the variable sharp_img
                # cv2.fliter2D() is the function used
                # src is the source of image(here, img)
                # depth is destination depth. -1 will mean output image will have same depth as input image
                # kernel is used for specifying the kernel operation (here, sharp_kernel)
                frame = cv2.filter2D(src=frame, ddepth=-1, kernel=sharp_kernel)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass

def generate_frames(frame):

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        faces = face_cascade.detectMultiScale(frame, 1.1, 7)
        smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
        # img = cv2.imread(img)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        for (x, y, w, h) in faces:
            # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 3)
            center = (x + w // 2, y + h // 2)
            frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (0, 0, 255), 3)

        return frame
    #     ret, buffer = cv2.imencode('.jpg', frame)
    #     frame = buffer.tobytes()
    # yield (b'--frame\r\n'
    #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.form.get('click') == 'Capture':
        global capture
        capture = 1
    elif request.form.get('grey') == 'Grey':
        global grey
        grey = not grey
    elif request.form.get('neg') == 'Negative':
        global neg
        neg = not neg
    elif request.form.get('Blur') == 'Blur':
        global Blur
        Blur = not Blur
    elif request.form.get('Invert') == 'Invert':
        global Invert
        Invert = not Invert
    elif request.form.get('sharp') == 'Sharpen':
        global sharp
        sharp = not sharp
    elif request.form.get('sepia') == 'Sepia':
        global sepia
        sepia = not sepia
    elif request.form.get('sketch') == 'Sketch':
        global sketch
        sketch = not sketch
    elif request.form.get('face') == 'Detect Face':
        global face
        face = not face
        if (face):
            time.sleep(4)
    elif request.form.get('stop') == 'Stop/Start':
        if (switch == 1):
            switch = 0
            camera.release()
            cv2.destroyAllWindows()
        else:
            camera = cv2.VideoCapture(0)
            switch = 1
    elif request.form.get('rec') == 'Start/Stop Recording':
        global rec, out
        rec = not rec
        if (rec):
            now = datetime.datetime.now()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
            # Start new thread for recording the video
            thread = Thread(target=record, args=[out, ])
            thread.start()
        elif (rec == False):
            out.release()
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='127.2.12.20', port=2000,debug=True)
