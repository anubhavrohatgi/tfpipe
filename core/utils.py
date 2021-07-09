import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from tfpipe.core.config import cfg
from ujson import load
from redis import Redis

from tfpipe.core.libs.tensorflow import resize  # <-- needed for namespace
# think about overriding filterboxes

##### GENERAL #####


def valid_extension(path):
    """ Returns True if `path` has a valid extension for an image. """

    return os.path.splitext(path)[-1].lower() in cfg.VALID_EXTS


def images_from_file(path: str, root: str = ""):
    """ Returns list of image paths from a file path. """

    if path.endswith('.json'):
        images = load(open(os.path.join(root, path), 'r'))
    else:
        images = [os.path.join(root, path)] if valid_extension(path) else []

    return images


def images_from_dir(path: str):
    """ Returns a list of paths to valid images. """

    if os.path.isdir(path):
        images = list()
        for root, _, files in os.walk(path):
            for img in files:
                if valid_extension(img):
                    images += images_from_file(img, root)
    else:
        images = images_from_file(path)

    return images


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

###################

##### TRAINING #####


def get_anchors(anchors, tiny=False):
    anchors = np.array(anchors)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)


def load_config(classes, model, is_tiny):
    if is_tiny:
        pass
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, is_tiny)

        XYSCALE = cfg.YOLO.XYSCALE if model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(classes))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE


def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'
        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

####################


##### EVALUATION #####

def create_redis(host, port):
    redis = Redis(host=host,
                  port=port,
                  db=0,
                  charset='utf-8',
                  decode_responses=True)

    return redis


def iter_bboxes(bboxes, num_classes):
    """ Yields the class indicator, bbox coordinates (TL, BR in (y, x)), and
    the score for each detection in bboxes. """

    out_boxes, out_scores, out_classes, num_boxes = [b[0] for b in bboxes]

    for i in range(num_boxes):
        class_ind = int(out_classes[i])
        if class_ind < 0 or class_ind > num_classes:
            continue

        coords = tf.split(out_boxes[i], 2)
        score = out_scores[i]

        yield class_ind, coords, score


def draw_bbox(image, bboxes, classes, show_label=True):
    """ Draws bounding boxes, and labels if show_label, on the image. """

    num_classes = len(classes)
    dims = tf.cast(image.shape[:2], tf.float32)

    hsv_tuples = [(1.0 * i / num_classes, 1., 1.) for i in range(num_classes)]
    colors = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for class_ind, coords, score in iter_bboxes(bboxes, num_classes):
        (y1, x1), (y2, x2) = tf.cast(dims * coords, tf.int32).numpy()

        bbox_color = colors[class_ind]
        bbox_thick = tf.cast(tf.reduce_sum(dims) / 1000, tf.int32).numpy()
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick)

        if show_label:
            bbox_mess = f"{classes[class_ind]}: {score:.2f}"
            t1, t2 = cv2.getTextSize(
                bbox_mess, 0, cfg.FONTSCALE, thickness=bbox_thick // 2)[0]
            c3 = (x1 + t1, y1 - t2 - 3)
            cv2.rectangle(image, (x1, y1), c3, bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.FONTSCALE, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def get_meta(shape, image_id, bboxes, classes):
    """ Returns the dection metadata as a list of dictionary entries. """

    num_classes = len(classes)
    image_h, image_w, *_ = shape

    out_boxes, out_scores, out_classes, num_boxes = [b[0] for b in bboxes]

    metadata = list()
    for i in range(num_boxes):
        class_ind = int(out_classes[i])
        if class_ind < 0 or class_ind > num_classes:
            continue

        coor = out_boxes[i]
        x1 = coor[1] * image_w
        x2 = coor[3] * image_w
        y1 = coor[0] * image_h
        y2 = coor[2] * image_h

        w = x2 - x1
        h = y2 - y1

        score = out_scores[i]

        d = {"image_id": image_id, "category_id": class_ind, "bbox": [
            x1, y1, w, h], "score": float(score)}

        # d = {"id": class_ind, "score": float(
        #     score), "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        metadata.append(d)

    return metadata

def convert_redis(file_path, shape, num_classes, bboxes):
    """ Converts predictions to the Redis output format. """

    dims = tf.cast(shape[:2], tf.float32)

    output = ""
    for class_ind, coords, score in iter_bboxes(bboxes, num_classes):
        (y1, x1), (y2, x2) = (dims * coords).numpy()

        output += f"{file_path},{class_ind},{x1},{y1},{x2},{y2},{score},"

    return output


def get_devices():
    """ Returns the GPUs and CPUs for the machine. """

    return (tf.config.list_physical_devices('GPU'),
            tf.config.list_physical_devices('CPU'))


def get_init_img(size):
    """ Returns `cfg.INIT_IMG` as a Tensor. """

    image = cv2.cvtColor(cv2.imread(cfg.INIT_IMG), cv2.COLOR_BGR2RGB)
    image = [cv2.resize(image, (size, size)) / 255.0]
    image = np.asanyarray(image).astype(np.float32)

    return tf.constant(image)


def filter_boxes(box_xywh, scores, input_shape, score_threshold=0.4):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    # print(box_xywh)
    class_boxes = tf.boolean_mask(box_xywh, mask)
    # print(class_boxes)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(
        scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(
        scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def fbox(box_xywh, scores, input_shape, score_threshold=0.4):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold

    # class_boxes = boolean_mask(box_xywh, mask)
    class_boxes = box_xywh
    # pred_conf = boolean_mask(scores, mask)
    pred_conf = scores
    class_boxes = tf.reshape(class_boxes, [tf.shape(
        scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(
        scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # boxes = boolean_mask(boxes, mask)
    # pred_conf = boolean_mask(pred_conf, mask)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (mask, boxes, pred_conf)


def build_preproc(size):
    """ Returns function used to preprocess an image. """

    spec = (tf.TensorSpec((size, size, 3), dtype=tf.dtypes.float32),)

    @ tf.function(input_signature=spec, jit_compile=True)
    def preproc(image):

        # Convert from BGR to RGB
        pp = tf.reverse(image, [-1])
        # pp = tf.image.resize(pp, (size, size)) / 255.0

        return tf.reshape(pp, (1, size, size, 3))

    return preproc


def build_predictor(framework, weights, size):
    """ Returns function used to make predictions. """

    if framework == 'tf':
        model = tf.keras.models.load_model(weights, compile=False)
        spec = (tf.TensorSpec((1, size, size, 3), dtype=tf.dtypes.float32),)

        @ tf.function(input_signature=spec, jit_compile=True)
        def predict(data):
            boxes, conf = model(data)

            mask, boxes, conf = fbox(boxes, conf, tf.constant([size, size]))

            return mask, boxes, conf

    elif framework == 'tflite':
        pass
    else:
        assert False, f"Invalid Framework: {framework}"

    return predict


######################
