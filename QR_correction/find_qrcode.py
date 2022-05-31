# 帶修正：
# 1. 調整光線(跟thresh有關)
# 2. 接收多大的圖片
# 3. 補插植
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyzbar.pyzbar as pyzbar
# import zxing
from PIL import ImageEnhance

# import decoder
import QR_correction.correction as correction

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # font

width, height = 0, 0

def main(img):

    if img.shape[0] < img.shape[1]:  # 假如長寬顛倒，就轉90度
        img = np.rot90(img)
    
    global width, height
    width, height = img.shape[1], img.shape[0]
    
    roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰階化

    roi_map = cv2.convertScaleAbs(roi_gray, alpha=1, beta=0)  # 將灰階轉換成黑白圖，增強對比度

    # roi_map = cv2.equalizeHist(roi_map)  # 對比度增強

    ret, roi_thresh = cv2.threshold(roi_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法閾值化

    # _, roi_thresh = cv2.threshold(roi_map, 80, 255, cv2.THRESH_BINARY)  # 閾值化

    roi_thresh = cv2.inpaint(roi_thresh, np.zeros(roi_thresh.shape, dtype=np.uint8), 3, cv2.INPAINT_TELEA)  # 填充黑色區域
    a, b = img.shape[:2]

    canvas = init_canvas(b+500, a+500)  # 初始化畫布

    canvas[250:250+a, 250:250+b] = roi_thresh  # 將圖片放入畫布中
    # cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)
    # cv2.imshow("canvas", canvas)
    # cv2.waitKey(0)

    qr_code = get_qrcode(canvas, ret=100)
    return qr_code


def imshow_rgb(title, img):  # 彩圖檢視
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def imshow_gs(title, img):  # 灰階顯示
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def init_canvas(width, height, color=(255)):  # 建立一張畫布，預設為白色
    canvas = np.ones((height, width), dtype="uint8")
    canvas[:] = color
    return canvas


def transform(image, pts, width, height):

    if width > height:  # 假如長寬顛倒，就交換
        width, height = height, width

    approx = np.float32(pts)
    pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(approx, pts)  # 取得仿射變換矩陣
    dst = cv2.warpPerspective(image, M, (int(width), int(height)))  # 仿射變換

    return dst


def getCorner(roi_edge, contours=None, pattern="outer"):  # 角點偵測
    global width, height
    if pattern == "outer":  # 假如是要抓外框
        
        # shi-tomasi corner detection
        corners = cv2.goodFeaturesToTrack(roi_edge, 4, 0.01, width//8, blockSize=10, useHarrisDetector=False, k=0.04)
        
        # subpixel accuracy
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        # cv2.cornerSubPix(roi_edge, np.float32(corners), (5, 5), (-1, -1), criteria)

        # 亞像素的角點偵測還在研究中
        
        corners = np.int0(corners)
        pts = []
        for c in corners:  # 將角點轉換成陣列
            x, y = c.ravel()
            pts.append([x, y])
            cv2.circle(roi_edge, (x, y), 3, (255), 3)

        pts = sort_points(pts)  # 排序
        return roi_edge, pts

    if pattern == "inner" and contours != None:
        for i in range(len(contours)):
            new_img = np.zeros(roi_edge.shape, dtype=np.uint8)
            cv2.drawContours(new_img, contours, i, (255, 255, 255), 1)  # 分別劃出每個輪廓

            corners = cv2.goodFeaturesToTrack(new_img, 4, 0.01, 10)
            corners = np.int0(corners)

            if len(corners) == 4:  # 如果可以抓到4的角點
                pts = []  # 將角點轉換成陣列
                for c in corners:
                    x, y = c.ravel()
                    pts.append([x, y])
                    cv2.circle((new_img), (x, y), 3, (255), 3)

                pts = sort_points(pts)  # 將角點重新按順時針排序

                if pts[1][1] - pts[0][1] <= 4 and pts[2][1] - pts[3][1] <= 4 and pts[3][0] - pts[0][0] <= 4 and pts[2][0] - pts[1][0] <= 4:
                    break
        roi_edge = 255-new_img
        module_x = max((pts[1][0] - pts[0][0])//7,
                       (pts[2][0] - pts[3][0])//7)
        module_y = max((pts[2][1] - pts[1][1])//7,
                       (pts[3][1] - pts[0][1])//7)

        return roi_edge, module_x, module_y


def sort_points(points):  # 將點重新按順時針排序
    if len(points) != 4:
        # print("Error: points number must be 4")

        return None
    else:
        points = sorted(points, key=lambda pt: pt[1])
        if points[0][0] > points[1][0]:
            points[0], points[1] = points[1], points[0]
        if points[3][0] > points[2][0]:
            points[3], points[2] = points[2], points[3]
        return points


def draw_approx_hull_polygon(img, contours):
    img = np.zeros(img.shape, dtype=np.uint8)
    hulls = [cv2.convexHull(cnt) for cnt in contours]  # 找輪廓的凸包
    cv2.polylines(img, hulls, True, 255, 1)
    # cv2.namedWindow("convex", cv2.WINDOW_NORMAL)
    # cv2.imshow("convex", img)
    # cv2.imwrite("convex.jpg", img)

    return img


def AreaBlack(img):  # 判斷黑色區域占多或是少
    black = 0
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if img[j, i] < 128:
                black += 1
    if black == 0:
        return 0

    return round(black/(img.shape[0]*img.shape[1]), 2)


def get_qrcode(img, ret=120):
    img_cp = img.copy()

    roi_blur = cv2.GaussianBlur(img_cp, (5, 5), 0)  # 高斯模糊

    roi_edge = cv2.Canny(roi_blur, 50, 70, apertureSize=3)

    # cv2.imshow("roi_edge", roi_edge)

    contours, _ = cv2.findContours(roi_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找輪廓
    # RETR_EXTERNAL: 只找外層輪廓
    
    # cp = roi_edge.copy()
    # for i in range(len(contours)):
    #     cv2.drawContours(cp, contours, i, (255), 1)  # 分別劃出每個輪廓
    # cv2.imshow("contour", cp)
    
    contours = [i for i in contours if cv2.contourArea(i) > cv2.arcLength(i, True)]  # 只留下閉合的輪廓
    # cv2.contourArea(i)：計算輪廓面積
    # cv2.arcLength(i, True)：計算輪廓長度，True表示關閉的輪廓

    contours = sorted(contours, key=lambda cnt: cv2.contourArea(cnt), reverse=True)  # 按面積排序
    # reverse=True：表示由大到小排序

    contours = [contours[0]]  # 只留下最大的輪廓

    roi_edge = draw_approx_hull_polygon(roi_edge, contours)  # 繪製凸包
    
    roi_edge, pts = getCorner(roi_edge, None)  # 取得角點
    
    # 在圖上畫出角點
    for (x, y) in pts:
        cv2.circle(roi_edge, (x, y), 3, (255), -1)
    # cv2.namedWindow("roi_edge", cv2.WINDOW_NORMAL)
    # cv2.imshow("roi_edge", roi_edge)

    height = pts[3][1] - pts[0][1]
    width = pts[2][0] - pts[3][0]

    dst = transform(img, pts, width, height)

    # dst = cv2.threshold(dst, 120, 255, cv2.THRESH_BINARY_INV)[1]  # 閾值處理
    # cv2.imshow("dst_before", dst)

    if round(dst.shape[0] / dst.shape[1]) <= 1:
        # print("Error: dst.shape[0] / dst.shape[1] <= 1")
        return None

    dst = correction.color_correct(dst)  # 將外圍的黑色邊框去除
    cv2.inpaint(dst, np.ones(dst.shape, dtype=np.uint8), 3, cv2.INPAINT_TELEA, dst)  # 補插值
    
    # for y in range(height):
    #     x = 0
    #     while dst[y, x] == 0:
    #         dst[y, x] = 255
    #         x+=1
    #     x = width-1
    #     while dst[y, x] == 0:
    #         dst[y, x] = 255
    #         x-=1
    
    # for x in range(width):
    #     y = 0
    #     while dst[y, x] == 0:
    #         dst[y, x] = 255
    #         y+=1
    #     y = height-1
    #     while dst[y, x] == 0:
    #         dst[y, x] = 255
    #         y-=1
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)

    # 右邊的留白可能會不夠，這種情況下就強行加白線在右邊
    dx = 100
    for y in range(dst.shape[0]):
        x = dst.shape[1]-1
        while dst[y, x] == 255 and x > 0:
            x -= 1
        if (dst.shape[1]-x-1) < dx:
            dx = dst.shape[1]-x-1
    if dx < 3:
        col = 255-np.zeros((dst.shape[0], 3-dx), dtype=np.uint8)
        dst = np.hstack((dst, col))

    # cv2.imshow("dst_after", dst)
    # cv2.waitKey(0)

    half_position = int(np.floor(dst.shape[0]/2))

    dst = np.where(dst < ret, 0, 255).astype("uint8")

    result = np.where(
        (dst == np.full(dst.shape[1], 255, dtype=np.uint8)).all(axis=1))[0]
    if len(result) < 3:
        return None

    a = result[np.where((result > half_position/2) &
                        (result < half_position*3/2))[0]]
    y1 = result[np.where(result < half_position/10)][-1]
    y4 = result[np.where(result > half_position*19/10)][0]
    # start, end = a[0], a[-1]
    y2, y3 = a[0], a[-1]

    half_1 = dst[y1:y2, :]
    half_2 = np.flip(dst[y3:y4, :], axis=0)
    # np.flip(dst, axis=0)：將矩陣上下翻轉

    half_1 = correction.color_correct(half_1)
    half_2 = correction.color_correct(half_2)
    

    half_2 = cv2.resize(half_2, (half_2.shape[1], half_1.shape[0]))
    s = sewing(half_1, half_2)
    return s


def sewing(half_1, half_2):
    half_1 = np.where(half_1 < 120, 0, 255).astype("uint8")
    half_2 = np.where(half_2 < 120, 0, 255).astype("uint8")

    check_reversed = False

    # cv2.namedWindow("half_2", cv2.WINDOW_NORMAL)
    # cv2.imshow("half_2", half_2)
    # 預防貼反：這裡如果有錯，那可能就是貼反了，可以嘗試上下翻轉，再事一次
    while True:
        try:
            # 寫法一：
            # for half_1
            x1 = half_1.shape[1] - 1
            for x in range(half_1.shape[1]-1, 0, -1):
                cnt_w = (half_1[:, x] == 255).sum()
                if cnt_w / half_1.shape[0] > 0.7:
                    x1 -= 1
                else:
                    break
            # for half_2
            x2 = 0
            for x in range(half_2.shape[1]-1):
                cnt_w = (half_2[:, x] == 255).sum()
                
                if cnt_w / half_2.shape[0] > 0.7:
                    x2 += 1
                else:
                    break
                
            
            # # 寫法二(我也不知道哪一種比較快)
            # result = np.where((half_1 == np.full((half_1.shape[0], 1), 255, dtype=np.uint8)).all(axis=0))[0]
            # for i in range(len(result)-1, 1, -1):
            #     if result[i]-1 != result[i-1]:
            #         x1 = result[i]
            #         break

            # result = np.where((half_2 == np.full((half_2.shape[0], 1), 255, dtype=np.uint8)).all(axis=0))[0]
            # for i in range(1, len(result)):
            #     if result[i]-1 != result[i-1]:
            #         x2 = result[i-1]
            #         break

            # 檢查：原本在組合的時候，給右半邊牆加了一條黑白相間的column，現在要把這條column去掉
            flag = 1
            while True:
                changePattern = 0
                for y in range(half_2.shape[0]-1):
                    if half_2[y, x2] < 120 and half_2[y+1, x2] > 125:
                        changePattern += 1
                if x2 >= half_2.shape[1]/2:
                    raise Exception
                # print(changePattern)
                if changePattern < 9 and flag == 0:
                    break
                if changePattern > 9:
                    if flag == 1:
                        flag = 0
                x2 += 1
            break
        except:
            if check_reversed:
                break
            half_1, half_2 = np.flip(half_2, axis=1), np.flip(half_1, axis=1)
            check_reversed = True

    # cv2.imshow('half1', half_1[:, :x1])
    # cv2.imshow('half2', half_2[:, x2:])

    qr_code = np.hstack((half_1[:, :x1], half_2[:, x2:]))
    # np.hstack(A, B)：將兩個矩陣水平組合
    
    space = round(qr_code.shape[0]/21)


    canvas = init_canvas(qr_code.shape[1]+8*space, qr_code.shape[0]+8*space)
    # init_canvas(width, height)：建立一張空白畫布
    canvas[4*space:qr_code.shape[0]+4*space, 4*space:qr_code.shape[1]+4*space] = qr_code
    # 將qr_code置中放入canvas

    cv2.imshow('QR_code', canvas)
    # reader = zxing.BarCodeReader()
    # texts = reader.decode("QR_code.jpg")
    texts = pyzbar.decode(canvas)
    flag = 0
    if texts == [] and flag == 0:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        texts = pyzbar.decode(canvas)
        flag = 1
        
    if texts == []:
        # print("未識別成功")
        # cv2.waitKey(0)
        return None
    else:
        for text in texts:
            tt = text.data.decode("utf-8")
            print(tt)
        # print("識別成功")
        # cv2.waitKey(0)
        return tt


if __name__ == "__main__":
    img = cv2.imread("tests\image07.jpg")

    main(img)
