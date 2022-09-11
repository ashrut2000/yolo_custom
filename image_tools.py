""" Image tools that perform small operation with images
"""
from types import FrameType
import numpy as np
from math import cos, sin, degrees

TO_DEGREE = 180/np.pi

def get_center(box):
    """Returns the center of the box given

    Args:
        box (tuple/list/array): bounding box, [x_min, y_min, x_max, y_max]

    Returns:
        center[array]: center of bounding box, [x,y]
    """
    return np.array(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), dtype=np.int16)


def get_bearing(point, frame_property=[640, 480, 64, 41], in_degree=True):
    """ Returns the bearing of the point with respect to image
        Up is positive and down is negative, right is positive and left is negative

    Args:
        point: [x, y], coordinates, of which depth is to be returned
        frame_property: [image_frame_width, image_frame_height, image_frame_view_angle_h, image_frame_view_angle_v]

    Returns:
        (x_angle, y_angle)[tuple with element in radian]: x and y bearing of given point, right-left-> pos-neg up-down-> pos-neg
    """    
    # viewing angle os the sensor in radian
    w, h, a_w, a_h = frame_property
    frame_ar = w/h
    camera_ar = a_w/a_h
    if frame_ar > camera_ar:
        a_h = a_h*(camera_ar/frame_ar)
    else:
        a_w = a_w*(frame_ar/camera_ar)
    print(a_w, a_h)
    frame_property[2] = a_w
    frame_property[3] = a_h

    viewangle_h = np.deg2rad(frame_property[2])
    viewangle_v = np.deg2rad(frame_property[3])

    point = ((point[0] - ( frame_property[0] / 2)),
             (point[1] - ( frame_property[1] / 2)))
    # print('point',point)

    angle_x = (point[0] * viewangle_h) / frame_property[0]
    angle_y = (point[1] * viewangle_v) / frame_property[1]
    if in_degree:
        return (angle_x*TO_DEGREE, angle_y*TO_DEGREE)
    return (angle_x, angle_y)

def polar_to_ordinates(bearing, distance):
    """ Converts the given polor coordinate to the ordinates (x, y) """
    b = bearing[0]
    d = distance

    x = cos(b) * d
    y = -sin(b) * d
    z = -sin(bearing[1]) * d

    return np.array((x, y, z), dtype=np.float)

def get_position_3D(box, distance, frame_property=[640, 480, 64, 41]):
    """Return th position of hte box in 3d , need depth information

    Args:
        box (numpy array): arrray
        depth_array ([type]): depth iamge
        frame_property(list)): dimension of image and view angles

    Returns:
        array: [x,y,z]
    """
    bearing = get_bearing(get_center(box), frame_property)
    # print("bearing", bearing)
    return polar_to_ordinates(bearing, distance)


def create_color_image(width=640, height=480, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB, opencv compatible

    Args:
        width (int, optional): width of image. Defaults to 640.
        height (int, optional): height of image. Defaults to 480.
        rgb_color (tuple, optional): color of image. Defaults to (0, 0, 0).

    Returns:
        image [opencv-matrix]: dummy image in bgr format, shape (height, width, channels)
    """
    # Create black blank image
    image = np.zeros((height, width, len(rgb_color)), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))

    # Fill image with color
    image[:] = color
    
    return image


def create_depth_image(width=640, height=480, default_value=255):
    """Create new 16-bitimage(numpy array) filled with certain intensity, opencv compatible

    Args:
        width (int, optional): width of image. Defaults to 640.
        height (int, optional): height of image. Defaults to 480.
        default_value (int, optional): iniial value of 16-bit image. Defaults to 255(white).

    Returns:
        image [opencv-matrix]: dummy 16-bit image grayscale format, shape (height, width, channels)
    """
    image = np.zeros((height, width, 1), np.uint16)
    image[:] = default_value
    return image


def test():
    """Tests the functions available in this module
    """
    print(get_center([10.5,20,20,30]))
    print(get_center([-10,-20,20,30]))
    print(create_color_image(200, 100,(200,100,3)).shape)
    print("Bearing of (1,1):",get_bearing((1,1)))
    import cv2
    frame = create_color_image(200, 100,(200,))
    frame_d = create_depth_image(200, 100, 65535)
    print(frame_d)
    cv2.imshow("frame", frame_d)
    cv2.waitKey(2000)


def associate_objects_to_faces(objects, faces):
    """Associate the detected person with detected faces, those with no face are left empty

    Args:
        objects (list): result from detection [label, box, score]
        faces (list): result from face detection [box,...]
    """
    
    def isBoxInside(box_small, box_big):
        """Detects if the box lies totally inside the bigger box"""
        # print(box_small, box_big)
        if (box_small[0] > box_big[0]) and (box_small[2] < box_big[2]) and (box_small[1] > box_big[1]) and (box_small[3] < box_big[3]):
            return True
        return False

    def get_area(box):
        """Returns the area of the box
        """
        return (box[2]-box[0])*(box[3]-box[1])

    # objects = [{"label": object[0], "bounding_box": object[1], "confidence": object[2]} for object in objects]
    pairs = objects
    if pairs and faces.any():
        sorted_objects = pairs.copy()
        sorted_objects.sort(key=lambda x: get_area(x["bounding_box"])) # 1 is the index of box
        for i in range(len(sorted_objects)):
            associated_faces = []
            for face in faces:
                if isBoxInside(face, objects[i]["bounding_box"]):
                    associated_faces.append(face)
            pairs[i]["face_box"] = associated_faces

    return pairs


def pair_items_from_bounding(objects, tracks):
    """Associates the tracker id with the detected objects on the basis of box proximity

    Args:
        objects (dict): list of dictionary of object and their properties, format: [{},{},{}]
        tracks (numpy_array): array of box and id given by the tracker, format: [x1,y1,x2,y2,tracking_id]
    """

    def check_similarity(box_1, box_2, tolerance = 5):
        """Checks the similarity between two boxes

        Args:
            box_1 (1darray): first box
            box_2 (1darray): second box
            tolerance (int, optional): pixel tolerance to fing similarity. Defaults to 5.

        Returns:
            [type]: [description]
        """
        for i in range(len(box_1)):
            if (box_1[0] < box_2[0] + tolerance) and (box_1[0] > box_2[0] - tolerance):
                continue
            else:
                return False
        return True

    object_tracked = []
    for track in tracks:
        box = track[:4]
        for object in objects:
            if check_similarity(object.get("bounding_box"), box):
                object["id"] = track[4]
                object_tracked.append(object)
                break
    
    return object_tracked


if __name__ == "__main__":
    print(get_bearing((440,400), frame_property=[640, 480, 64, 41]))
    print(get_bearing((440,360), frame_property=[640, 480, 64, 41]))
    print(get_bearing((160,360), frame_property=[640, 480, 64, 41]))
    print(get_bearing((160,120), frame_property=[640, 480, 64, 41]))