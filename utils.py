import numpy as np
import math
from scipy.signal import savgol_filter
import cv2
def add_depth(detections, use_depth=True):
    # for a in data_sequence:
    result = np.empty((len(detections), 6))

    for i in range(len(result)):
        box = detections[i]['bounding_box']
        result[i][0] = box[0]
        result[i][1] = box[1]
        result[i][2] = 0
        result[i][3] = box[2]
        result[i][4] = box[3]
        if(detections[i].get('depth')) and use_depth:
            result[i][5] = detections[i]['depth']
        else:
            result[i][5] = 1

    return result



def get_velocity(box_current, box_previous):

    x_current = (box_current[0]+box_current[3])/2
    y_current = (box_current[1]+box_current[4])/2
    x_prev = (box_previous[0]+box_previous[3])/2
    y_prev = (box_previous[1]+box_previous[4])/2
    return x_current-x_prev, y_current-y_prev

def box_area(box): #2D
    width= box[2]-box[0]
    height=box[3]-box[1]
    return width*height

def depth_scale(detections, frame, max_depth=15 ):
    frame_height=frame.shape[0]
    frame_width= frame.shape[1]
    frame_area=frame_width*frame_height
    print('frame_area',frame_area)
    slope= (max_depth-0)/(0-1)
    #slope=1
    for i in range(len(detections)):
        box=detections[i]['bounding_box']
        area_of_box=box_area(box)    
        ratio= area_of_box/frame_area
        depth= max_depth + slope* ratio
        #depth=-(slope*ratio)
        #depth=depth/3
        detections[i]['depth']=depth
        print('box area ','bounding box','depth', area_of_box, box,depth)
    return detections





def get_velocity_endpoint(n1,n2,v_x1,v_y1):
    v_x1 = v_x1 + 0.00000000001 # small value is added so that we dont get infinity
    v_y1 = v_y1 + 0.00000000001
    v_length=(math.sqrt(v_x1**2+v_y1**2))
    if(v_length>40):
        v_length=40
    if(v_x1>0 and v_y1<0):
        theta=(math.atan(v_y1/v_x1))
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)
    elif(v_x1<0 and v_y1<0):
        theta=math.atan(v_y1/v_x1)-math.pi
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)
    elif(v_x1<0 and v_y1>0):
        theta=math.pi+(math.atan(v_y1/v_x1))
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)
    else:
        theta=math.atan(v_y1/v_x1)
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)

def get_box_center(box):
    x= (box[0]+box[3])/2
    y= (box[1]+box[4])/2
    return x,y

def average_box_data(previous,current):
    for i in range(len(previous)):
        current[i]=(0.2*current[i]+0.8*previous[i])
        return current
def draw_arrowed_line(frame, center, end_point):
    cv2.arrowedLine(frame, center, end_point, (255, 0, 0), 5)
    return frame

def filter_data(xc,yc,xp,yp):
    length=len(xp)
    l=-length
    xc=np.array(xc[l:])
    yc=np.array(yc[l:])
    xp=np.array(xp[l:])
    yp=np.array(yp[l:])
    if(length%2==0):
        length=length-1
        return savgol_filter(xc, length, 4)[-1], savgol_filter(yc, length, 4)[-1],savgol_filter(xp, length, 4)[-1], savgol_filter(yp, length, 4)[-1]
    else:
        return savgol_filter(xc, length, 4)[-1], savgol_filter(yc, length, 4)[-1],savgol_filter(xp, length, 4)[-1], savgol_filter(yp, length, 4)[-1]

def filter_setup( current_tracked_detections,previous_tracked_detections,x_current,y_current,x_previous,y_previous,visual):
    visual1=visual.copy()

    for i in current_tracked_detections:
            if i in previous_tracked_detections:
                box_current = current_tracked_detections[i]['bounding_box']
                box_previous = previous_tracked_detections[i]['bounding_box']
                x1, y1 = get_box_center(box_current)
                x11,y11=get_box_center(box_previous)
                if i not in x_current:
                    x_current[i]=[]
                if i not in x_previous:
                    x_previous[i]=[]
                if i not in y_current:
                    y_current[i]=[]
                if i not in y_previous:
                    y_previous[i]=[]
                x_current[i].append(x1)
                y_current[i].append(y1)
                x_previous[i].append(x11)
                y_previous[i].append(y11)
                
                if((len(x_current[i])>5)):
                    
                    filtered_x_current,filtered_y_current,filtered_x_previous,filtered_y_previous=filter_data(x_current[i],y_current[i],x_previous[i],y_previous[i])
                    vx,vy=(filtered_x_current-filtered_x_previous),(filtered_y_current-filtered_y_previous)
                    x1, y1 = get_box_center(box_current)
                    x2, y2 = get_velocity_endpoint(filtered_x_current, filtered_y_current, vx*30, vy*30)
                    visual1=draw_arrowed_line(visual1, center=(int(filtered_x_current), int(filtered_y_current)),end_point=(int(x2), int(y2))) 
                else:
                    box_current_plot=average_box_data(previous_tracked_detections[i]['bounding_box'],current_tracked_detections[i]['bounding_box'])
                    vx, vy = get_velocity(box_current_plot, box_previous)
                    x1, y1 = get_box_center(box_current)
                    x2, y2 = get_velocity_endpoint(x1, y1, vx*30, vy*30)
                    visual1=draw_arrowed_line(visual1, center=(int(x1), int(y1)),end_point=(int(x2), int(y2))) 
                    

            else:
                visual1=visual1
    return visual1


if __name__ == '__main__':
    from detectionss import DetectionYOLO
    weights='./checkpoints/yolov4-416'
    Detect=DetectionYOLO(weights)
    try:
        frame=cv2.imread('image500.jpg')
        detections, visual=Detect.detect(frame,ret=True)
        #print('detections', detections)
        detections=depth_scale(detections,visual)
        #print('detections2', detections)
        cv2.imshow('visual', visual)
        cv2.waitKey(0)



        
    except SystemExit:
        pass
