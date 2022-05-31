
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyzbar.pyzbar as pyzbar
from PIL import ImageEnhance

# import decoder
import QR_correction.correction as correction


def main(img):
    if img.shape[0] < img.shape[1]:
        img = np.rot90(img)

    ratio = img.shape[0] / img.shape[1]

    if round(ratio) < 2:  # 長/寬的比例小於2
        return None

    roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale image

    roi_map = cv2.convertScaleAbs(
        roi_gray, alpha=1, beta=0)   # 將灰階轉換成黑白圖，增強對比度
    
    # roi_map = cv2.equalizeHist(roi_map)  # 將圖片直方圖均衡化
    
    _, roi_thresh = cv2.threshold(roi_map, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 大津法閾值化
    
    # _, roi_thresh = cv2.threshold(roi_map, 80, 255, cv2.THRESH_BINARY)  # 普通閾值化
    
    canvas = np.full((img.shape[0]+500, img.shape[1]+500), 255, dtype=np.uint8)  # 建立一張白色的圖片
    
    canvas[250:-250, 250:-250] = roi_thresh  # 將閾值化後的圖片放入canvas
    
    message = get_qrcode(canvas)  # 取得QR code
    
    return message

def transform(image, pts, width, height):

    if width > height:  # 假如長寬顛倒，就交換
        width, height = height, width

    approx = np.float32(pts)
    pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(approx, pts)  # 取得仿射變換矩陣
    dst = cv2.warpPerspective(image, M, (int(width), int(height)))  # 仿射變換

    return dst


def get_qrcode(img):
    copy = img.copy()
    
    roi_blur = cv2.GaussianBlur(copy, (5, 5), 0)  # 高斯模糊
    
    roi_edge = cv2.Canny(roi_blur, 50, 70, apertureSize=3)  # Canny邊緣檢測
    
    contours, _ = cv2.findContours(roi_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 取得輪廓
    # RETR_EXTERNAL: 只找外層輪廓
    
    contours = [i for i in contours if cv2.contourArea(i) > cv2.arcLength(i, True)]  # 取得閉合的輪廓
    # cv2.contourArea(i)：計算輪廓面積
    # cv2.arcLength(i, True)：計算輪廓長度，True表示關閉的輪廓
    
    contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)  # 按面積排序
    # reverse=True：表示由大到小排序
    
    contours = [contours[0]]  # 只留下最大的輪廓
    
    roi_edge = draw_approx_hull_polygon(roi_edge, contours)  # 繪製凸包
    
    pts = find_Corner(roi_edge)  # 取得角點
    
    height = max(pts[3][1], pts[2][1])-min(pts[0][1], pts[1][1])  # 取得圖片長寬
    width = max(pts[1][0], pts[2][0])-min(pts[0][0], pts[3][0])
    
    dst = transform(img, pts, width, height)  # 轉換圖片
    
    # cv2.imshow("dst_before", dst)

    if round(dst.shape[0] / dst.shape[1]) <= 1:
        return None
    
    
    
    


def draw_approx_hull_polygon(img, contours):
    img = np.zeros(img.shape, dtype=np.uint8)
    
    hulls = [cv2.convexHull(cnt) for cnt in contours]  # 找輪廓的凸包
    
    cv2.polylines(img, hulls, True, 255, 1)
    
    return img

def find_Corner(img):
    # shi-tomasi corner detection
    corners = cv2.goodFeaturesToTrack(img, maxCorners=4, qualityLevel=0.01, minDistance=10, blockSize=10, useHarrisDetector=False)
    # maxCorners: 最大角點數目
    # qualityLevel: 角點質量門檻值(小於1.0的正數，一般在0.01-0.1之間)
    # minDistance: 最小距離，小於此距離的點忽略
    # blockSize: 表示在計算角點時參與運算的區域大小，常用值為3，但是如果影象的解析度較高則可以考慮使用較大一點的值
    # useHarrisDetector: 用於指定角點檢測的方法，如果是true則使用Harris角點檢測，false則使用Shi Tomasi演算法
    
    # 亞畫素角點檢測
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    # cv2.cornerSubPix(roi_edge, np.float32(corners), (5, 5), (-1, -1), criteria)
    
    if len(corners) < 4:
        raise Exception("Not enough corners")
    
    corners = np.int0(corners)
    pts = []
    for c in corners:
        x, y = c.ravel() # 將角點轉換成一維陣列
        pts.append([x, y])
        cv2.circle(img, (x, y), 3, 255, -1)
        
    # cv2.imshow("corners", img)
    
    # 0 ----- 1
    # 
    # 3 ----- 2
    pts = sorted(pts, key=lambda coords: coords[1])  # 按y軸排序
    if pts[0][0] > pts[1][0]:
        pts[0], pts[1] = pts[1], pts[0]
    if pts[3][0] > pts[2][0]:
        pts[2], pts[3] = pts[3], pts[2]
        
    return pts