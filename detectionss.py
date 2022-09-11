# In this module we perform object detection task
# import the necessary packages
import numpy as np
from PIL import Image
import imutils
import cv2
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.python.saved_model import tag_constants
import time
import core.utils as utils1
from core.yolov4 import filter_boxes
from core.functions import *
tf.device("cpu:0")



class DetectionYOLO:

    # def __init__(self,video_path,weights,iou_threshold=0.45,score_threshold=0.5):
    def __init__(self, weights, iou_threshold=0.45, score_threshold=0.5):

        # self.video_path=video_path
        self.weights = weights
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        # self.video_name = video_path.split('/')[-1]
        # self.video_name = self.video_name.split('.')[0]
        self.saved_model_loaded = tf.saved_model.load(
            weights, tags=[tag_constants.SERVING]
        )
        self.infer = self.saved_model_loaded.signatures["serving_default"]
        # print(self.saved_model_loaded.signatures["input_1"])
        # print(self.saved_model_loaded.layers)
        # # print(dir(self.saved_model_loaded))
        # exit(0)
        self.input_size = 416
        self.frame_num = 0

    def detect(self, image, ret):
        detections = []
        frame = image.copy()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_num += 1
            image = Image.fromarray(frame)
        else:
            print("Video has ended or failed, try a different video format!")
            exit()

        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils1.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [
            bboxes,
            scores.numpy()[0],
            classes.numpy()[0],
            valid_detections.numpy()[0],
        ]
        w, h = 4, valid_detections.numpy()[0]
        dets = [[0 for x in range(w)] for y in range(h)]
        for i in range(valid_detections.numpy()[0]):
            dets[i] = bboxes[i]
        for i in range(valid_detections.numpy()[0]):
            detections.append({"bounding_box": dets[i], "score": scores.numpy()[0][i]})

        class_names = utils1.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        #allowed_classes=['person']
        image = utils1.draw_bbox(
            frame,
            pred_bbox,
            info=False,
            allowed_classes=allowed_classes,
            read_plate=False,
        )
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return detections, result

    def test(self, src=0):
        """test the operation of detection class

        Args:
            src (int, optional): video stream index if camera is attached. Defaults to 0.
        """
        from imutils.video import VideoStream

        vs = VideoStream(src=src).start()
        while 1:
            frame = vs.read()
            label, image_tag = self.detect(frame, True)
            # image_tag = frame
            # self.visualize(image_tag)
            cv2.imshow("detections", image_tag) 
            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    weights = "./checkpoints/yolov4-416"
    obj = DetectionYOLO(weights)
    image, label = obj.test()
    # print(image, label)
