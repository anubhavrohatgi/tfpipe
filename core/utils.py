import os
import cv2
import random
import colorsys
import numpy as np
from core.config import cfg
from ujson import load

def valid_extension(path):
    """ Returns True if `path` has a valid extension for an image. """
    
    return os.path.splitext(path)[-1] in cfg.VALID_EXTS

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


def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * i / num_classes, 1., 1.) for i in range(num_classes)]
    colors = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = [b[0] for b in bboxes]
    for i in range(num_boxes):
        class_ind = int(out_classes[i])
        if class_ind < 0 or class_ind > num_classes:
            continue

        coor = out_boxes[i]
        coor[0] = coor[0] * image_h
        coor[2] = coor[2] * image_h
        coor[1] = coor[1] * image_w
        coor[3] = coor[3] * image_w
        

        score = out_scores[i]
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(
                bbox_mess, 0, cfg.FONTSCALE, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (c3[0],
                                      c3[1]), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.FONTSCALE, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                        
    return image


def get_meta(shape, bboxes, classes):
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

        score = out_scores[i]

        d = {"id": class_ind, "score": float(score), "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        metadata.append(d)
    
    return metadata

def convert_redis(file_path, shape, num_classes, bboxes):
    """ Converts predictions to the Redis output format. """

    image_h, image_w, *_ = shape

    out_boxes, out_scores, out_classes, num_boxes = [b[0] for b in bboxes]

    output = ""

    for i in range(num_boxes):
        class_ind = int(out_classes[i])
        if class_ind < 0 or class_ind > num_classes:
            continue
        
        coor = out_boxes[i]
        x1 = coor[1] * image_w
        x2 = coor[3] * image_w
        y1 = coor[0] * image_h
        y2 = coor[2] * image_h

        score = out_scores[i]

        output += f"{file_path},{class_ind},{x1},{y1},{x2},{y2},{score},"
    
    return output