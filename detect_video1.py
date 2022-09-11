import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tracker3d import Tracker3D
from utils import filter_setup, get_box_center
from ImplementFilter import SavgolFilter
from absl import app, flags, logging
import core.utils as utils1
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from detectionss import DetectionYOLO
import math
from image_tools import get_bearing , polar_to_ordinates


def plot_id(tracked_data_current_frame, frame, show_id):
    new_frame = frame.copy()
    if(show_id):
        for i in tracked_data_current_frame:
            box_current = tracked_data_current_frame[i]["bounding_box"]
            cv2.putText(new_frame, str(
                i), (int(box_current[0]+5), int(box_current[1]+5)), 0, 0.5, (0, 250, 0), 2)
    return new_frame


def draw_arrowed_line(frame, center, end_point):
    frame=cv2.arrowedLine(frame, center, end_point, (255, 0, 0), 5)
    return frame
def draw_detections(display_id,details,current_tracked_detections):
    frame=display_id.copy()
    count=0
    print('lenght', len(details))
    for i in details:
        if i in current_tracked_detections:

            center,end_points=details[i][-1][0],details[i][-1][1]
            center=(center[0],center[1])
            end_points=(end_points[0],end_points[1])
            print('id','cetner', center,i)
            if(math.dist(center,end_points)<45):
                frame=draw_arrowed_line(frame,center,end_points)
                count+=1
    print('count',count)
    return frame

def animation(img,current_tracked_detections, details):
    frame=img.copy()
    height,width,channel=frame.shape
    for id, det in current_tracked_detections.items():
        # id=i
        box=det['bounding_box']
        box_center=get_box_center(box)
        center=(int(box_center[0]),int(box_center[1]))
        bearing=get_bearing(center, frame_property=[width, height, 64, 41])
        distance=box[5]
        ordinate=polar_to_ordinates(bearing, distance)
        wy,wz,wx= ordinate
        print('box','ordinate',box,ordinate)
        det_center_p= (int((height/5)*wx), int((height/10)*wy))
        camera_position=(int(width/2),height)
        #det_center_wrt_cam = (-det_center_p[0] +camera_position[0], -det_center_p[0] + camera_position[1])
        det_center_wrt_cam = (center[0], -det_center_p[1] + camera_position[1])
        d = det_center_wrt_cam
        # det_pos= (int(center[0]+camera_position[0]), int(center[1]+camera_position[1]))
        cv2.circle(frame, center=d, radius=30, color=(id*10,id*20,id*30), thickness=-1)
        cv2.circle(frame, center=camera_position, radius=30, color=(0,255,0), thickness=-1)
        cv2.putText(frame, str(id), d, 0, 0.5, (0, 250, 0), 2)
        # if id in details:
        #     end_point=details[id][-1][1]
        #     draw_arrowed_line(frame, d, end_point)
    return frame

def tracker_representation(img, current_tracked_detections, details):
    frame=img.copy()
    height,width,channel=frame.shape
    for id, det in current_tracked_detections.items():
        box=det['bounding_box']
        box_center=get_box_center(box)
        center=(int(box_center[0]),int(box_center[1]))
        camera_position=(int(width/2),height)
        cv2.circle(frame, center, radius=30, color=(id*10,id*20,id*30), thickness=-1)
        cv2.circle(frame, center=camera_position, radius=30, color=(0,255,0), thickness=-1)
        cv2.putText(frame, str(id), center, 0, 0.5, (0, 250, 0), 2)
        if id in details:
             end_point=details[id][-1][1]
             draw_arrowed_line(frame, center, end_point)
    return frame



def main(_argv):
    
    config = ConfigProto()
    input_size = 416
    video_path="./data/video/test_video_2.webm"
    fps=30
    weights='./checkpoints/yolov4-416'
    img = np.zeros((540,960,3), np.uint8)
    top_image=np.zeros((540*2,960,3), np.uint8)

    # get video name by using split method
    iou_threshold=0.45
    score_threshold=0.5
    Detect=DetectionYOLO(weights,iou_threshold,score_threshold)
    Tracker = Tracker3D()
    filter=SavgolFilter()
    

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    frame_num = 0
    while True:
        detections=[]
        ret, frame = vid.read()
        height,width, channel= frame.shape
        print('h','w',height,width)
        #iprint('h','w',height,width)f frame_num==0:
            #img = np.zeros((height,width,3), np.uint8)
        t1 = time.monotonic()
        detections, visual=Detect.detect(frame,ret)
        current_tracked_detections, previous_tracked_detections = Tracker.update(
            detections,frame)
        display_id = plot_id(current_tracked_detections, visual, show_id=True)
        details=filter.update(current_tracked_detections)
        final=draw_detections(display_id,details,current_tracked_detections)
        animation_image=animation(img,current_tracked_detections, details)
        animation_image=cv2.resize(animation_image, (width,height*2), interpolation = cv2.INTER_AREA)
        tracking_image= tracker_representation(img, current_tracked_detections, details)
        img_2=np.concatenate((final,tracking_image), axis=0)
        print('h',img_2.shape[0],img_2.shape[1])
        img_3 = np.concatenate((img_2,animation_image), axis=1)
        print('h1',img_3.shape[0],img_3.shape[1])
        new=cv2.resize(img_3, (int(img_3.shape[1]*0.6),int(img_3.shape[0]*0.6)), interpolation = cv2.INTER_AREA)
        print('h2',new.shape[0],new.shape[1])
        cv2.imshow("final", new) 
         
        
           
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
