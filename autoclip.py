import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import tensorflow as tf
import numpy as np

import config as cfg
from tfpipe.core.config import cfg as tpcfg
from tfpipe.core.utils import get_init_img, images_from_dir, read_class_names, build_preproc, build_predictor, images_from_dir, draw_bbox
from tfpipe.pipeline.pipeline import Pipeline

from time import time, time_ns

from multiprocessing import set_start_method, Process, Queue, SimpleQueue, Manager, Value
from multiprocessing.managers import BaseManager
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import ray
from ray.util.queue import Queue as rayq
from ray.util.queue import Empty

from collections import deque

from ptdiag import PTDiag, PTProcess
from time import sleep

import json

BATCH_SIZE = 30 * cfg.MODEL.AUTOCLIP.CLIP_DURATION
Q_LIMIT = 3000 / BATCH_SIZE
VIDEO_PATH = ""
WEIGHTS = ["checkpoints/trt-t4-float16", "checkpoints/trt-t4-float16", "checkpoints/trt-t4-float16", "checkpoints/trt-t4-float16", "checkpoints/trt-p6-float32", "checkpoints/trt-p6-float32", "checkpoints/trt-p6-float32", "checkpoints/trt-p6-float32"]
# WEIGHTS = ["checkpoints/trt-titanv-fp16"] * 2
OUTPUT_PATH = ""

def image_input(output_q, iminp_id, num_inputs, ptp_id):
    ptp = PTProcess(f"image input: {iminp_id}", ptp_id)

    ## Disable GPUs for processes other than predict
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")

    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = iminp_id * frames // num_inputs
    stop = (iminp_id + 1) * frames // num_inputs
    batch_id = start // BATCH_SIZE
    print(f"image input: {iminp_id} | start: {start} | stop: {stop} | batch_id: {batch_id}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)


    working = True
    ptp.start()
    while working:
        ptp.on()
        output_batch = list()
        for _ in range(BATCH_SIZE):
            ret, frame = cap.read()

            if not ret or start == stop:
                cap.release()
                working = False
                break
            
            output_batch.append((start, frame))
            start += 1

        if len(output_batch) > 0:
            # print(ptp._name, batch_id, start)
            ref = ray.put((batch_id, output_batch))
            ptp.off()
            output_q.put(ref)
            batch_id += 1

    ptp.off()
    print(ptp._name + " done")
            

def preproc_img(input_q, output_q, ptp_id):
    ptp = PTProcess(f"image preproc: {ptp_id}", ptp_id)

    ## Disable GPUs for processes other than predict
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")

    preproc = build_preproc(672)
    test_image = cv2.imread(tpcfg.INIT_IMG)
    test_image = tf.image.resize(test_image, (672, 672)) / 255.0
    test_image = preproc(test_image)

    ptp.start()
    while True:
        try:
            data = input_q.get(block=False)
        except Empty:
            continue

        # shm = SharedMemory(name=batch)
        # batch = np.ndarray((BATCH_SIZE, 720, 1280, 3), dtype=np.int8, buffer=shm.buf)
        
        ptp.on()
        
        if data is Pipeline.Exit:
            break

        batch_id, batch = data
        output_batch = [(frame_id, frame, preproc(tf.image.resize(frame, (672, 672)) / 255.0))
                            for frame_id, frame in batch]
        
        ref = ray.put((batch_id, output_batch))
        ptp.off()
        output_q.put(ref)

        # shm.close()
        # output_q.put(output_batch)

    
    ptp.off()
    print(ptp._name + " done")
    

def predict(input_q, output_q, device_id, ptp_id, ready, go):
    ptp = PTProcess(f"predict @ gpu: {device_id}", ptp_id)

    gpu = tf.config.list_physical_devices("GPU")[device_id]
    tf.config.set_visible_devices([gpu], "GPU")
    tf.config.experimental.set_memory_growth(gpu, True)
    gpu_cfg = [tf.config.LogicalDeviceConfiguration(memory_limit=5000)]
    tf.config.set_logical_device_configuration(gpu, gpu_cfg)
    vgpu = tf.config.list_logical_devices("GPU")[0]
    
    # with tf.device(vgpu.name):
    print(f"GPU: {device_id} | Loading Model...")
    weights = WEIGHTS[device_id]
    predict, model = build_predictor("tf", weights, 672, True)

    print(f"GPU: {device_id} | Inferencing Test Image...")
    predict(get_init_img(672))
    ready.value = 1
    
    print(f"GPU: {device_id} | Finished warmup, waiting for signal to start...")
    while go.value != 1:
        pass

    print(f"GPU: {device_id} | Beginning Processing...")
    ptp.start()
    while True:
        try:
            data = input_q.get(block=False)
        except Empty:
            continue

        ptp.on()
        
        if data == Pipeline.Exit:
            break
        
        batch_id, batch = data
        output_batch = [(frame_id, frame, predict(pp)) for frame_id, frame, pp in batch]
        ref = ray.put((batch_id, output_batch))
        ptp.off()
        output_q.put(ref)
        # output_q.put(output_batch)

    
    ptp.off()
    print(ptp._name + " done")


def annotate(input_q, output_q, ptp_id):
    ptp = PTProcess(f"annotate: {ptp_id}", ptp_id)
    
    ## Disable GPUs for processes other than predict
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")

    model_classes = read_class_names("data/.names")
    
    ptp.start()
    while True:
        try:
            data = input_q.get(block=False)
        except Empty:
            continue

        ptp.on()

        if data == Pipeline.Exit:
            break

        batch_id, batch = data
        output_batch = list()
        for frame_id, frame, pred in batch:
            boxes = pred["tf.reshape_9"]
            scores = pred["tf.reshape_10"]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes,
                    scores,
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=0.45,
                    score_threshold=0.25
            )

            dok = bool(valid_detections[0])

            output_batch.append((frame_id, frame, dok))
        
        ref = ray.put((batch_id, output_batch))
        ptp.off()
        output_q.put(ref)
        # output_q.put(output_batch)

    
    ptp.off()
    print(ptp._name + " done")


def meta_write(input_q, output_q, ptp_id):
    ptp = PTProcess(f"meta writer: {ptp_id}", ptp_id)

    ## Disable GPUs for processes other than predict
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")

    ## Output Directory
    os.makedirs("output", exist_ok=True)
    
    clip_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    # meta_path = os.path.join("scripts", "clip-meta", f"{clip_name}-predseg2-metadata.json")
    meta_path = os.path.join(OUTPUT_PATH, f"{clip_name}-predseg-metadata.json")

    metadata = dict()

    ptp.start()
    while True:
        try:
            data = input_q.get(block=False)
        except Empty:
            continue
        
        ptp.on()
        
        if data == Pipeline.Exit:
            break
        
        batch_id, batch = data
        num_pos = sum([1 for *_, dok in batch if dok])
        is_pos = num_pos / BATCH_SIZE >= cfg.MODEL.AUTOCLIP.DETECT_PERCENT

        for frame_id, frame, dok in batch:
            metadata[frame_id] = is_pos
        
        ref = ray.put((batch_id, batch, is_pos))
        ptp.off()
        output_q.put(ref)

    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    
    ptp.off()
    print(ptp._name + " done")


def video_write(input_q, ptp_id):
    ptp = PTProcess(f"video writer: {ptp_id}", ptp_id)

    ## Disable GPUs for processes other than predict
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")

    ## Output Directories
    clip_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    pos_output_dir = os.path.join(OUTPUT_PATH, clip_name + "-pos")
    neg_output_dir = os.path.join(OUTPUT_PATH, clip_name + "-neg")
    os.makedirs(pos_output_dir, exist_ok=True)
    os.makedirs(neg_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    resolution = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ptp.start()
    while True:
        try:
            data = input_q.get(block=False)
        except Empty:
            continue
            
        ptp.on()

        if data == Pipeline.Exit:
            break
        
        batch_id, batch, is_pos = data
            
        output_dir = pos_output_dir if is_pos else neg_output_dir
        path = os.path.join(output_dir, f"{clip_name}-{batch_id}.mp4")
        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)
        
        for _, frame, _ in batch:
            video.write(frame)
        
        video.release()
        ptp.off()
    
    ptp.off()
    print(ptp._name + " done")
        

##############################################


##############################################


if __name__ == '__main__':
    ptd = PTDiag()
    ptp_id = 0
    ptp = PTProcess("main", ptp_id, exclude_from_graph=True)
    ptp_id += 1
    # ptp.on()

    ## MAX EFFICIENCY NEEDS: num_inputs == num_outputs
    num_inputs = 6
    num_preprocs = 8
    num_predicts = 4 #8
    num_annotaters = 6
    num_meta = 1
    num_writers = 4

    ## MP Start Method
    set_start_method("spawn", force=True)

    ## Disable GPUs for processes other than predict
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([], "GPU")

    ## Allocate Memory
    # print("Allocating memory...")
    # shms = malloc()

    preproc_q = rayq(Q_LIMIT)
    pred_q = rayq(Q_LIMIT)
    anno_q = rayq(Q_LIMIT)
    meta_q = rayq(Q_LIMIT)
    write_q = rayq(Q_LIMIT)


    # iminp_refs = np.array_split(output, 3)
    inputs = list()
    for i in range(num_inputs):
        p = Process(target=image_input, args=(preproc_q, i, num_inputs, ptp_id), daemon=True)
        p.start()
        inputs.append(p)
        ptp_id += 1

    preprocs = list()
    for i in range(num_preprocs):
        p = Process(target=preproc_img, args=(preproc_q, pred_q, ptp_id), daemon=True)
        p.start()
        preprocs.append(p)
        ptp_id += 1

    predictors = list()
    gpu_ready_flags = list()
    go = Value('i', 0)
    for i in range(num_predicts):
        ready = Value('i', 0)
        p = Process(target=predict, args=(pred_q, anno_q, i, ptp_id, ready, go), daemon=True)
        p.start()
        predictors.append(p)
        gpu_ready_flags.append(ready)
        ptp_id += 1

    annotaters = list()
    for i in range(num_annotaters):
        p = Process(target=annotate, args=(anno_q, meta_q, ptp_id), daemon=True)
        p.start()
        annotaters.append(p)
        ptp_id += 1

    meta_writers = list()
    for i in range(num_meta):
        p = Process(target=meta_write, args=(meta_q, write_q, ptp_id), daemon=True)
        p.start()
        meta_writers.append(p)
        ptp_id += 1

    writers = list()
    for i in range(num_writers):
        p = Process(target=video_write, args=(write_q, ptp_id), daemon=True)
        p.start()
        writers.append(p)
        ptp_id += 1


    # GPU Start Together
    while not all([v.value == 1 for v in gpu_ready_flags]):
        pass
    
    print("All GPUs ready...")
    go.value = 1
    ptp.on()

    while any([p.is_alive() for p in inputs]):
        pass
    
    print("inputs done")
    for _ in preprocs:
        # preproc_q.put(Pipeline.Exit)
        preproc_q.put(ray.put(Pipeline.Exit))
    
    while any([p.is_alive() for p in preprocs]):
        pass

    print("preprocs done")
    for _ in predictors:
        # pred_q.put(Pipeline.Exit)
        pred_q.put(ray.put(Pipeline.Exit))
    
    while any([p.is_alive() for p in predictors]):
        pass
    
    print("preds done")
    for _ in annotaters:
        # anno_q.put(Pipeline.Exit)
        anno_q.put(ray.put(Pipeline.Exit))

    while any([p.is_alive() for p in annotaters]):
        pass

    print("annos done")
    for _ in meta_writers:
        meta_q.put(ray.put(Pipeline.Exit))
    
    while any([p.is_alive() for p in meta_writers]):
        pass
    
    print("meta writers done")
    for _ in writers:
        write_q.put(ray.put(Pipeline.Exit))

    while any([p.is_alive() for p in writers]):
        pass
    
    print("writers done")

    ## Free Memory
    # print("Freeing shared memory...")
    # for shm in shms:
    #     shm.close()
    #     shm.unlink()

    ptp.off()
    ptd.graph_all(save=True)
    # print(ptd)
    overall_output = "\n\n*** DID NOT FILL IN OVERALL OUTPUT ***\n\n"
    s = ""
    for name, ptp_id, num_edges, time_on, time_off, rate_on, rate_off in ptd.get_stats():
        time_on, time_off, rate_on, rate_off = np.around(
                [time_on/1e9, time_off/1e9, rate_on*1e9*BATCH_SIZE, rate_off*1e9*BATCH_SIZE], decimals=2)
        
        if name == "main":
            cap = cv2.VideoCapture(VIDEO_PATH)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            rate_on = num_frames / time_on
            video_dur = num_frames / fps / 60 # minutes
            time_on = round(time_on / 60, 3)
            overall_output = f"\n\n*** OVERALL PERFORMANCE :"
            overall_output += f" | Video Duration: {video_dur} min, Processing Runtime: {time_on} min"
            overall_output += f" | {num_frames} frames @ {rate_on} fps ***\n\n"
        else:
            s += f"Name: {name} & PTP ID: {ptp_id} -- {num_edges*BATCH_SIZE} edges"
            s += f" | {time_on} s on, {time_off} s off"
            s += f" | {rate_on} e/s on, {rate_off} e/s off\n"
    
    print(s)
    print(overall_output)