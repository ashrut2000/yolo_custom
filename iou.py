import numpy as np
def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  z_value=1
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  zz1=z_value
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  zz2=z_value
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  intersection_vol= w * h*z_value
  o = intersection_vol / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])*z_value                                     
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])*z_value - wh*z_value)                                               
  return(o)

if __name__ == '__main__':
    o=iou_batch([1,2,3,4],[2,3,4,5])
    print(o)