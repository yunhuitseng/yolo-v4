import argparse
import os
import random
import time
from collections import OrderedDict
from ctypes import *
from glob import glob
# from multiprocessing import Process, Pool
# import multiprocessing as mp
from queue import Empty, Queue
from threading import Thread, enumerate

import cv2
import numpy as np
import QR_correction.find_qrcode as qrcode
from pyimagesearch.centroidtracker import CentroidTracker

from darknet.build.darknet.x64 import darknet

darknet_height, darknet_width = 0, 0
video_width, video_height = 0, 0
cap = None
args = ""
network, class_names, class_colors = None, None, None
fps = None
lock = True
qrcode_list = OrderedDict()

def transform(image, location=None):

    x, y, w, h = location
    width, height = w, h
    approx = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])

    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(approx, pts2)
    dst = cv2.warpPerspective(image, M, (int(width), int(height)))
    return dst


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="tests/video02.mp4",
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="darknet/build/darknet/x64/deron/backup/yolov4-tiny-custom_2000.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="darknet/build/darknet/x64/deron/yolov4-tiny-custom.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="darknet/build/darknet/x64/deron/deron.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert4cropping(image, bbox):
    global darknet_height, darknet_width
    x, y, w, h = darknet.convert2relative(bbox, darknet_height, darknet_width)

    image_h, image_w, __ = image.shape

    orig_left = int((x - w / 2.) * image_w)
    orig_right = int((x + w / 2.) * image_w)
    orig_top = int((y - h / 2.) * image_h)
    orig_bottom = int((y + h / 2.) * image_h)

    if (orig_left < 0):
        orig_left = 0
    if (orig_right > image_w - 1):
        orig_right = image_w - 1
    if (orig_top < 0):
        orig_top = 0
    if (orig_bottom > image_h - 1):
        orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

lock2 = True
def video_capture(frame_queue, darknet_image_queue):
    global video_width, video_height, cap, lock2
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            _ = darknet_image_queue.get()
            darknet_image_queue.put(None)
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    lock2 = False
    # darknet_image_queue.put(None)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    global network, class_names, class_colors, cap
    while cap.isOpened():
        # if darknet_image_queue.empty():
        #     break
        try:
            darknet_image = darknet_image_queue.get(False)
            # print(darknet_image)
            prev_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
            detections_queue.put(detections)
            t_v = time.time()-prev_time
            if t_v == 0:
                fps = 27
            else:
                fps = int(1/t_v)
            fps_queue.put(fps)
            # print("FPS: {}".format(fps))
            # darknet.print_detections(detections, args.ext_output)
            darknet.free_image(darknet_image)
        except Empty as e:
            if(lock2):
                # print("done inference 1")
                continue
            # print("done inference 2")
            break
    # print("done inference 3")
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue, img_for_decode, read_before):
    random.seed(3)  # deterministic bbox colors
    global cap, fps, qrcode_list, lock
    fourcc = cv2.VideoWriter_fourcc('D', 'I','V', 'X')
    video = set_saved_video(cap, "output.avi",(video_width, video_height))
    ct = CentroidTracker()
    
    while True:
        
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        
        if frame is not None:
            rects = []
            
            for label, confidence, bbox in detections:
                if float(confidence) < 93: # 當信心指數小於95時，挑過去
                    continue
                bbox_adjusted = darknet.convert2original(frame, bbox, darknet_height, darknet_width)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
                x, y, w, h = bbox_adjusted

                rects.append((int(round(x - (w / 2))), int(round(y - (h / 2))), int(round(x+w/2)), int(round(y+h/2))))

            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)

            objects, bounds = ct.update(rects)
            
            
            for ((objectID, centroid), (x, y, w, h)) in zip(objects.items(), bounds.values()):
                text = "ID {}".format(objectID)
                if objectID not in read_before:
                    try:
                        qrcode_list[objectID] = "unknown"
                        img_for_decode.put((transform(image=frame, location=(x, int(y-h/45*2), w, int(h+h/45*4))), objectID))
                    except:
                        break
                
                if x < 0:
                    continue
                
                text += ": " + qrcode_list[objectID]
                x1, y1 = x, y
                color = class_colors['QRcode']
                text_color = (255-color[0], 255-color[1], 255-color[2])

                # 計算文字大小
                (wt, ht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                # 加入文字背景區塊
                cv2.rectangle(image, (x1, y1 - ht - 10), (x1 + wt, y1), color, -1)
                # 加入文字
                cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                
                
                
            cv2.namedWindow("Inference", cv2.WINDOW_NORMAL)
            cv2.imshow('Inference', image)
            video.write(image)
            if cv2.waitKey(fps) == 27:
                break
        if frame_queue.qsize() < 4:
            break
    print("done")
    cv2.destroyWindow('Inference')
    lock = False
    cap.release()
    video.release()


def decode_frame(img_for_decode, read_before):
    global cap, fps, qrcode_list, lock
    # first = True
    while cap.isOpened():
        # if(first):
        #     frame, objectID = img_for_decode.get(False)
        #     first = False
        try:
            frame, objectID = img_for_decode.get(False)
            if objectID in read_before:
                img_for_decode.queue.clear()
                continue
            if frame is not None and lock:
                cv2.imshow('temp', frame)
                text = qrcode.main(frame)
                
                if text is not None:
                    read_before.append(objectID)
                    qrcode_list[objectID] = text
                    print(qrcode_list)
                    cv2.destroyWindow('temp')
                
                if cv2.waitKey(fps) == 27:
                    break
        except:
            continue
    cv2.destroyWindow('QR_code')
    cap.release()


def main(argcs):
    global cap, darknet_height, darknet_width, video_width, video_height, args, lock, lock2
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=5)
    fps_queue = Queue(maxsize=5)
    
    img_for_decode = Queue()
    read_before = []
    
    args = argcs
    global network, class_names, class_colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)


    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 這兩句是重複多次運行的命脈
    lock = True
    lock2 = True
    
    th_list = []
    th_list.append(Thread(target=video_capture, args=(frame_queue, darknet_image_queue)))
    
    th_list.append(Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)))
    
    th_list.append(Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, img_for_decode, read_before)))
    
    th_list.append(Thread(target=decode_frame, args=(img_for_decode, read_before)))
    
    [t.start() for t in th_list]
    [t.join() for t in th_list]
    
    global qrcode_list
    
    
    with open('output.txt', 'w') as f:
        for key, value in qrcode_list.items():
            f.write(str(key) + " " + str(value) + "\n")
    
    return qrcode_list

if __name__ == '__main__':
    main()
