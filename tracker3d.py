
from filters import KalmanBoxTracker
import argparse
# import time
# import glob
# from skimage import io
# import matplotlib.patches as patches
#import matplotlib.pyplot as plt
from tkinter import *
# import os
import numpy as np
#import matplotlib

from utils import add_depth,get_velocity, depth_scale
# matplotlib.use('TkAgg')

#from mina.cognition.utility.decorator import time_span
np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,z1,x2,y2,z2]
    """

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    zz1 = np.maximum(bb_test[..., 2], bb_gt[..., 2])
    xx2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    yy2 = np.minimum(bb_test[..., 4], bb_gt[..., 4])
    zz2 = np.minimum(bb_test[..., 5], bb_gt[..., 5])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    d = np.maximum(0., zz2 - zz1)
    wh = w*h
    intersection_vol = w * h*d

    # o = intersection_vol/ ((bb_test[..., 3] - bb_test[..., 0]) * (bb_test[..., 4] - bb_test[..., 1])* (bb_test[..., 5] - bb_test[..., 2])
    #   + (bb_gt[..., 3] - bb_gt[..., 0]) * (bb_gt[..., 4] - bb_gt[..., 1])*(bb_gt[..., 5] - bb_gt[..., 2])-intersection_vol)
    o = wh / ((bb_test[..., 3] - bb_test[..., 0]) * (bb_test[..., 4] - bb_test[..., 1])
              + (bb_gt[..., 3] - bb_gt[..., 0]) * (bb_gt[..., 4] - bb_gt[..., 1]) - wh)
    return(o)


def get_matched_indices(detections, trackers, iou_threshold=0.3):
    """ Associates the detection to the existing objects in the tracker
    Args:
        detections [list]:  list of numpy arrays
        trackers [list]: list of existing array
        iou_threshold:  """
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)

        else:
            matched_indices = linear_assignment(-iou_matrix)

    else:
        matched_indices = np.empty(shape=(0, 2))
    return matched_indices


class Tracker3D(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.tracked_data_previous_frame = {}
        self.tracked_data_current_frame={}
        
        
    

    def _update(self, dets=np.empty((0, 7))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,z1,x2,y2,z2,score],[x1,y1,z1,x2,y2,z2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 7))  # m

        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], 0, pos[3], pos[4], 1, 0]

            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            dets, trks, self.iou_threshold)
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            d[2]=0
            print('d',d)


            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 7))  # m

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """

        if(len(trackers) == 0):
            # m
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 7), dtype=int)

        iou_matrix = iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)

            else:
                matched_indices = linear_assignment(-iou_matrix)

        else:
            matched_indices = np.empty(shape=(0, 2))
    

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:, 0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if(t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if(len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    
    def update(self, detections,frame):
        self.tracked_data_previous_frame=self.tracked_data_current_frame
        self.tracked_data_current_frame = {}
        dets=depth_scale(detections,frame)
        dets = add_depth(dets)
        trackers = self._update(dets)
        matched_indices = get_matched_indices(dets, trackers)
        for m in matched_indices:
            detections[m[0]]['bounding_box']=trackers[m[1]]
        for m in matched_indices:
            self.tracked_data_current_frame[trackers[m[1]][6]]=detections[m[0]]
        return self.tracked_data_current_frame, self.tracked_data_previous_frame
