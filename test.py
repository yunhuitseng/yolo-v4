# from logging import critical
# import cv2
# import numpy as np

# src = cv2.imread("convex.jpg")
# img = src.copy()
# copy = src.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# corners = cv2.goodFeaturesToTrack(thresh, 4, 0.01, 10, blockSize=10, useHarrisDetector=False, k=0.04)
# corners = np.int0(corners)

# for c in corners:
#     (x, y) = c.ravel()
#     print(c)
#     src = cv2.circle(src, (x, y), 10, (255, 0, 255), -1)

# cv2.namedWindow("corners", cv2.WINDOW_NORMAL)
# cv2.imshow("corners", src)
# cv2.waitKey()

# # src = copy.copy()

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
# cv2.cornerSubPix(thresh, np.float32(corners), (5, 5), (-1, -1), criteria)
# for c in corners:
#     (x, y) = c.ravel()
#     print(c)
#     src = cv2.circle(src, (x, y), 10, (0, 0, 255), -1)

# cv2.namedWindow("corners", cv2.WINDOW_NORMAL)
# cv2.imshow("corners", src)
# cv2.waitKey()


# import cv2
# import pyzbar.pyzbar as pyzbar

# src = cv2.imread("QR_code.jpg")
# src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# blurred = cv2.GaussianBlur(src, (1, 1), 0)
# # th = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# # _, th = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# text = pyzbar.decode(blurred)
# print(text)
# # print(text[0].data.decode("utf-8"))

# cv2.namedWindow("src", cv2.WINDOW_NORMAL)
# cv2.imshow("src", blurred)
# cv2.waitKey()

import random

import cv2
import numpy as np

from user_app import makecode_single as makecode

for z in range(100):
    
    canvas = np.zeros((1920, 1080), dtype="uint8")

    module = 5

    for y in range(0, 1920, module):
        for x in range(0, 1080, module):
            color = random.randint(0, 1)*255
            canvas[y: y+module, x: x+module] = color

    for i in range(3):
        x = module*random.randint(0, 1080//module)
        y = module*random.randint(0, 1920//module)

        error_code = makecode.error_code()
        if x+error_code.shape[1] > 1080:
            error_code = error_code[:, : -(x+error_code.shape[1] - 1080)]
        if y+error_code.shape[0] > 1920:
            error_code = error_code[: -(y+error_code.shape[0] - 1920), :]

        canvas[y:y+error_code.shape[0], x:x+error_code.shape[1]] = error_code


        qrcode = makecode.make_code("13456")

        x = module*random.randint(0, 1080//module)
        y = module*random.randint(0, 1920//module)

        if x+qrcode.shape[1] > 1080:
            qrcode = qrcode[:, : -(x+qrcode.shape[1] - 1080)]
        if y+qrcode.shape[0] > 1920:
            qrcode = qrcode[: -(y+qrcode.shape[0] - 1920), :]


        canvas[y:y+qrcode.shape[0], x:x+qrcode.shape[1]] = qrcode

    # cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
    # cv2.imshow("canvas", canvas)
    # cv2.waitKey()
    cv2.imwrite("testcase/test"+str(z)+".jpg", canvas)
