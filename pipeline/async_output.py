
import cv2
import tensorflow as tf

from multiprocessing import Process, Queue

from tfpipe.core.utils import read_class_names, draw_bbox
from tfpipe.pipeline.pipeline import Pipeline

class AsyncOutput(Pipeline):
    """ Pipeline task for asynchronous output. """

    class _Worker(Process):
        def __init__(self, args, output_queue):
            self.iou_thresh = args.iou
            self.score_thresh = args.score
            self.classes = read_class_names(args.classes)
            self.output_queue = output_queue
            
            super().__init__(daemon=True)

        
        def run(self):
            tf.config.set_visible_devices([], "GPU")
            with tf.device("CPU:0"):
                while True:
                    data = self.output_queue.get()
                    if data == Pipeline.Exit:
                        break

                    boxes, scores = data["predictions"]

                    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                        boxes,
                        scores,
                        max_output_size_per_class=50,
                        max_total_size=50,
                        iou_threshold=self.iou_thresh,
                        score_threshold=self.score_thresh
                    )

                    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                                valid_detections.numpy()]

                    annotated_image = draw_bbox(
                        data["image"].copy(), pred_bbox, self.classes)

                    cv2.imshow("Output", annotated_image)
                    cv2.waitKey(1)
                    # cv2.imwrite(data["image_path"], annotated_image)

    def __init__(self, args):
        self.output_queue = Queue()

        self._worker = self._Worker(args, self.output_queue)

        self._worker.start()

        super().__init__()
    
    def map(self, data):
        self.output_queue.put(data)
        
        return data

    def cleanup(self):
        """ Kills all predictors. """

        self.output_queue.put(Pipeline.Exit)