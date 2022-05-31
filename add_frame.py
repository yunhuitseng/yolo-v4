import cv2
import os
cap = cv2.VideoCapture("tests/video06.mov")
if os.path.exists("frames/video06/") == False:
    os.mkdir("frames/video06/")
else:
    for f in os.listdir("frames/video06/"):
        filename = os.path.join("frames/video06/", f)
        if os.path.isfile(filename):
            try:
                os.remove(filename)
            except OSError as e:
                print(e)
cnt = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite("frames/video06/{}.jpg".format(cnt), frame)
    cnt += 1
cap.release()
