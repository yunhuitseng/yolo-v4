import argparse
import os
from queue import Queue

import cv2

import darknet.darknet_images as dn
import darknet.darknet_video as dv
import QR_correction.find_qrcode as qrcode

# def examine_file():
#     # 去download找檔案路徑
#     Path = "C:\Users\ketty\Downloads\shelf.webcam"
#     with open(Path, "r") as f:
#         if os.path.isfile(Path):
            

def parser(type="image", path=""):
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument("--weights", default="darknet/build/darknet/x64/deron/backup/yolov4-tiny-custom_2000.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="darknet/build/darknet/x64/deron/yolov4-tiny-custom.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="darknet/build/darknet/x64/deron/deron.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    if type == "image":
        # image source
        parser.add_argument("--input", type=str,
                            default=path)

        # number of images to be processed at the same time
        parser.add_argument("--batch_size", default=1, type=int)

        parser.add_argument("--ext_output", action='store_true',
                            help="display bbox coordinates of detected objects")
        parser.add_argument("--save_labels", action='store_true',
                            help="save detections bbox for each image in yolo format")
    if type == "video":
        parser.add_argument("--input", type=str, default=path)

        # inference video name. Not saved if empty
        parser.add_argument("--out_filename", type=str, default="")
        # display bbox coordinates of detected objects
        parser.add_argument("--ext_output", action='store_true')
    return parser.parse_args()


def main(filepath="tests/result/image27.jpg"):
    args = parser(path=filepath)

    img_for_decode = dn.main(args)

    while not img_for_decode.empty():
        img = img_for_decode.get()
        try:
            qrcode.main(img)
            cv2.waitKey()
        except:
            continue


def video(filepath): 
    args = parser("video", path=filepath)
    qr_list = dv.main(args)
    # for key, value in qr_list.items():
    #     print(value)
    return qr_list


if __name__ == '__main__':
    
    main()
    # video(filepath="tests/video06.mov")
    # video(filepath="video_upload/1650790684645.mp4")
