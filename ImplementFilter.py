import numpy as np
from utils import get_box_center, average_box_data, filter_data, get_velocity_endpoint,get_velocity
class SavgolFilter:
  def __init__(self):
    self.x_current={}
    self.x_previous={}
    self.y_current={}
    self.y_previous={}
    self.details={}
    self.previous_data={}

  def update(self,current_tracked_detections):
    for i in current_tracked_detections:
        if i in self.previous_data:
            box_current = current_tracked_detections[i]['bounding_box']
            box_previous = self.previous_data[i]['bounding_box']
            x1, y1 = get_box_center(box_current)
            x11,y11=get_box_center(box_previous)
            if i not in self.x_current:
                self.x_current[i]=[]
            if i not in self.x_previous:
                self.x_previous[i]=[]
            if i not in self.y_current:
                self.y_current[i]=[]
            if i not in self.y_previous:
                self.y_previous[i]=[]
            if i not in self.details:
                self.details[i]=[]
            self.x_current[i].append(x1)
            self.y_current[i].append(y1)
            self.x_previous[i].append(x11)
            self.y_previous[i].append(y11)
            if((len(self.x_current[i])>5)):
                
                filtered_x_current,filtered_y_current,filtered_x_previous,filtered_y_previous=filter_data(self.x_current[i],self.y_current[i],self.x_previous[i],self.y_previous[i])
                vx,vy=(filtered_x_current-filtered_x_previous),(filtered_y_current-filtered_y_previous)
                x1, y1 = get_box_center(box_current)
                x2, y2 = get_velocity_endpoint(filtered_x_current, filtered_y_current, vx*30, vy*30)
                center=[int(filtered_x_current),int(filtered_y_current)]
                end_points=[int(x2),int(y2)]
                self.details[i].append([center,end_points])
            else:
                box_current_plot=average_box_data(self.previous_data[i]['bounding_box'],current_tracked_detections[i]['bounding_box'])
                vx, vy = get_velocity(box_current_plot, box_previous)
                x1, y1 = get_box_center(box_current)
                x2, y2 = get_velocity_endpoint(x1, y1, vx*30, vy*30)
                center=[int(x1),int(x2)]
                end_points=[int(x2),int(y2)]
                self.details[i].append([center,end_points]) 
            self.previous_data=current_tracked_detections
        self.previous_data=current_tracked_detections
    return self.details




