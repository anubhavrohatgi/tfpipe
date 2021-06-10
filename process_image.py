import os
from multiprocessing import set_start_method

from core.config import cfg

from pipeline.pipeline import Pipeline
from pipeline.image_input import ImageInput
# from pipeline.async_predict import AsyncPredict
from pipeline.predict import Predict
from pipeline.annotate_image import AnnotateImage
from pipeline.image_output import ImageOutput

from time import time

def parse_args():
    """ Parses command line arguments. """

    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(
        description="TensorFlow YOLOv4 Image Processing Pipeline")
    ap.add_argument("-w", "--weights", required=True,
                    help="path to weights file")
    ap.add_argument("-i", "--input", required=True,
                    help="path to the input image/directory or list of file paths stored in a json file")
    ap.add_argument("-s", "--size", type=int, default=416,
                    help="the value to which the images will be resized")

    # Model Settings
    ap.add_argument("-f", "--framework", default="tf",
                    help="the framework of the model")
    ap.add_argument("--tiny", action="store_true",
                    help="use yolo-tiny instead of yolo")
    ap.add_argument("--iou", default=0.45, help="iou threshold")
    ap.add_argument("--score", default=0.25, help="score threshold")
    ap.add_argument("--classes", default=cfg.YOLO.CLASSES, help="file path to classes")

    # Output Settings
    ap.add_argument("-o", "--output", default="output",
                    help="path to the output directory")
    ap.add_argument("--full-output-path", action="store_true",
                    help="saves output to path including its respective input's full basename")
    ap.add_argument("--show", action="store_true",
                    help="display image after prediction")

    # Mutliprocessing settings
    ap.add_argument("--gpus", type=int, default=1,
                    help="number of GPUs (default: 1)")
    ap.add_argument("--cpus", type=int, default=0,
                    help="number of CPUs (default: 1)")
    ap.add_argument("--queue-size", type=int, default=3,
                    help="queue size per process (default: 3)")
    ap.add_argument("--single-process", action="store_true",
                    help="force the pipeline to run in a single process")

    return ap.parse_args()


def main(args):
    """ The main function for image processing. """

    # Create output directory if needed
    os.makedirs(args.output, exist_ok=True)

    # Image output type
    output_type = "vis_image"

    # Create pipeline tasks
    image_input = ImageInput(path=args.input)

    if not args.single_process and False:
        set_start_method("spawn", force=True)
        predict = AsyncPredict(num_gpus=args.gpus,
                               num_cpus=args.cpus,
                               queue_size=args.queue_size,
                               ordered=False)
    else:
        predict = Predict(args.weights, args.framework, size=args.size)

    annotate_image = AnnotateImage(output_type, args.iou, args.score, args.classes)

    # temp image output for testing
    image_output = ImageOutput(output_type, args)

    # Create the image processing pipeline
    pipeline = (image_input
                >> predict
                >> annotate_image
                >> image_output)

    # Main Loop
    t = time()
    results = list()
    while image_input.is_working() or predict.is_working():
        if (result := pipeline.map(None)) != Pipeline.Empty:
            results.append(result)
            
    print(f"Runtime: {time() - t} s")

    predict.cleanup()
    # print("Results: " + str(result))


if __name__ == '__main__':
    args = parse_args()
    main(args)
