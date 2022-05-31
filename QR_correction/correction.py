import math
import random

import cv2
import numpy as np
import sys
sys.setrecursionlimit(15000)
cnt = 0


def flood_recursive(matrix, color):
    width = len(matrix)
    height = len(matrix[0])
    global cnt

    def fill(x, y, start_color, color_to_update):
        global cnt
        if cnt > 3000:
            return
        # if the square is not the same color as the starting point
        if matrix[x][y] != start_color:
            return
        # if the square is not the new color
        elif matrix[x][y] == color_to_update:
            return
        else:
            # update the color of the current square to the replacement color
            matrix[x][y] = color_to_update
            cnt += 1
            neighbors = [(x-1, y), (x+1, y), (x-1, y-1), (x+1, y+1),
                         (x-1, y+1), (x+1, y-1), (x, y-1), (x, y+1)]
            for n in neighbors:
                if 0 <= n[0] <= width-1 and 0 <= n[1] <= height-1:
                    fill(n[0], n[1], start_color, color_to_update)
    corner = [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]]

    for (start_x, start_y) in corner:

        if matrix[start_x][start_y] != color:
            fill(start_x, start_y, matrix[start_x][start_y], color)
    return matrix


def transform(image, pts, width, height):

    if width > height:  # 假如長寬顛倒，就交換
        width, height = height, width

    approx = np.float32(pts)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(approx, pts2)  # 取得仿射變換矩陣
    dst = cv2.warpPerspective(image, M, (int(width), int(height)))  # 仿射變換

    return dst

# print(flood_recursive(m))


def turn_HSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 100])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('mask1', mask)

    lower = np.array([0, 0, 150])
    upper = np.array([180, 30, 255])
    mask2 = cv2.inRange(hsv, lower, upper)
    cv2.imshow('mask2', mask2)

    return 255-(mask | 255-mask2)


def color_correct(img):

    img = np.asarray(img)
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)  # 閾值化

    height, width = img.shape[:2]
    start_point = [(0, 0), (0, height-1), (width-1, height-1), (width-1, 0)]

    for pt in start_point:
        cv2.floodFill(img, None, pt, (255, 255, 255))
    
    return img

def color_correct_inv(img):

    img = np.asarray(img)
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)  # 閾值化

    height, width = img.shape[:2]
    start_point = [(0, 0), (0, height-1), (width-1, height-1), (width-1, 0)]

    for pt in start_point:
        cv2.floodFill(img, None, pt, (255, 255, 255))
    
    return img

def double_linear(image, zoom_multiples=1):
    '''
    雙線性插值
    :param input_signal: 輸入圖像
    :param zoom_multiples: 放大倍數
    :return: 雙線性插值後的圖像
    '''
    img_copy = np.copy(image)   # 輸入圖像的副本

    height, width = image.shape[:2]  # 輸入圖像的尺寸（行、列）

    for j in range(1, height-1):
        for i in range(1, width-1):
            # 輸出圖片中座標 （i，j）對應至輸入圖片中的最近的四個點點（x1，y1）（x2, y2），（x3， y3），(x4，y4)的均值
            image[j, i] = (img_copy[j+1, i]-img_copy[j-1, i])\
                + (img_copy[j, i+1]-img_copy[j-1, i])\
                + (img_copy[j-1, i-1]+img_copy[j-1, i]-img_copy[j, i+1]-img_copy[j+1, i])\
                + (img_copy[j-1, i])

    return image


def find_cycle(img):  # 回傳[回]字的圖
    pass


def getbinary(qrcode):
    img = qrcode.copy()

    np.set_printoptions(threshold=np.inf)

    path = '1.txt'
    f = open(path, 'w')
    for row in img:
        f.write(str(row)+"\n")
    f.close()

    with open('2.txt', 'w') as f:
        with open('1.txt', 'r') as fp:
            for line in fp:
                if line[-2] != ']':
                    line = str(line).replace("\n", "")
                f.write(line)


if __name__ == '__main__':
    half1 = cv2.imread('half1.png', 0).astype(np.float)
    half2 = cv2.imread('half2.png')

    _, half1 = cv2.threshold(half1, 127, 255, cv2.THRESH_BINARY)
    out = double_linear(half1, 1).astype(np.uint8)
    getbinary(out)
    cv2.imshow('out', out)
    cv2.imwrite('out.png', out)
    cv2.waitKey()
