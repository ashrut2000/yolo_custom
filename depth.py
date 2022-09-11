"""Depth estimation tools
Note: histogram based depth estimator has minimum and maximum constraint
    also this method fails with blank images (need noise in data)
"""
import cv2
import numpy as np

# from .decorator import time_span
from .image_tools import get_center


def _point_depth(box, depth_array, default=-1):
    '''Returns the depth of the point from depth_array
    If depth is less than 0.3 meter then the depth is invalid'''
    point = get_center(box)
    # print("points: ", point)
    try:
        depth = depth_array[point[1], point[0]] / 1000
    except (IndexError):
        depth = default
    # if depth < 0.3:
    #     depth = -1
    return depth


def _multi_point_depth(box, depth_array, samples=5, scale=0.5):
    """return depth from combination of sample points"""
    dx = abs(box[2] - box[0])
    dy = abs(box[3] - box[1])

    range_x = np.array((box[0]+dx*scale/2, box[2]-dx*scale/2), dtype=np.int)
    range_y = np.array((box[1]+dy*scale/2, box[3]-dy*scale/2), dtype=np.int)

    if (np.diff(range_x) < 0) or (np.diff(range_y) < 0):
        print("scale should be less than or equal to 1")
        return None

    else:
        sample_points = [[np.random.randint(
            *range_x), np.random.randint(*range_y)] for i in range(samples)]

    depth = [depth_array[p[1], p[0]] for p in sample_points]
    return min(depth) / 1000


def _get_hist_depth(box, depth_array):
    """returns the histogram and depth of the ROI in given depth image
        mimimum depth = 0.3 meter
    Args:
        box (array<int>): [x1, y1, x2, y2]
        depth_array (2-d image array): Grayscale image 16-bit
    Returns:
        depth(in meter): depth of ROI in image is returned respectively
    """
    def apply_filter(input):
        # application of filter
        filtered = input
        return filtered
    # print(depth_array.shape)
    assert len(depth_array.shape) == 2 and len(box) == 4

    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    # print(box)
    depth_array_roi = depth_array[y1:y2, x1:x2]
    # depth_array_roi = depth_array[x1:x2, y1:y2]

    depth_array_blur = cv2.GaussianBlur(depth_array_roi, (3, 3), 3)

    depth_array_hist = np.histogram(
        depth_array_blur, bins="auto", range=(300, 8000))

    # print(depth_array_hist)
    hist = (depth_array_hist[0], depth_array_hist[1][0:-1])
    # print(hist)
    val = max(hist[0])
    # print(val)
    index = np.where(hist[0] == val)
    depth = hist[1][index][0]

    return depth / 1000 # return value in meter


def get_depth(box, depth_array, method=1):
    '''Returns the depth of the point in meter from depth_array using the method specified
    Methods:
        1: direct depth from depth image
        2: use multi point minimum depth calculation
        3: Uses histogram to measre depth form depth image, min 0.3 meter'''
    depth_array = np.array(depth_array)
    if method == 1:
        return _point_depth(box, depth_array) / 1000
    elif method == 2:
        return _multi_point_depth(box, depth_array) / 1000
    elif method == 3:
        return _get_hist_depth(box, depth_array) / 1000

def get_depth_estimator(method=1):
    '''Returns the depth of the point in meter from depth_array using the method specified
    Methods:
        1: direct depth from depth image
        2: use multi point minimum depth calculation
        3: Uses histogram to measre depth form depth image, min 0.3 meter'''
    if method == 1:
        return _point_depth
    elif method == 2:
        return _multi_point_depth
    elif method == 3:
        return _get_hist_depth

if __name__ == "__main__":
    from image_tools import create_color_image, create_depth_image

    depth_image = create_depth_image(200, 100, 1000)
    # print(depth_image)
    print(get_depth([23,1,25,6], depth_image, 1))
    print(get_depth([23,1,25,6], depth_image, 2))
    print(get_depth([23,1,25,6], depth_image, 3))