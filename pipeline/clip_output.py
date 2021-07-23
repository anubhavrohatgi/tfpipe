import os
import cv2

from tfpipe.pipeline.pipeline import Pipeline

class ClipOutput(Pipeline):
    """ Pipeline task the outputs each frame from a clip to seperate videos 
    depending on the detection of an object in the frame."""

    def __init__(self, output_dir, video_ext=".mp4"):
        self.output_dir = output_dir
        self.video_ext = video_ext
        self.vid_pos = self.vid_neg = None

        super().__init__()

    def prep_output(self, path, fps, resolution):
        clip_name = os.path.splitext(os.path.basename(path))[0]

        pos_output = os.path.join(self.output_dir, f"{clip_name}-pos{self.video_ext}")
        neg_output = os.path.join(self.output_dir, f"{clip_name}-neg{self.video_ext}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.vid_pos = cv2.VideoWriter(pos_output, fourcc, fps, resolution)
        self.vid_neg = cv2.VideoWriter(neg_output, fourcc, fps, resolution)

    def write_clips(self):
        self.vid_pos.release()
        self.vid_neg.release()

    def map(self, data):
        if data["detection"]:
            self.vid_pos.write(data["image"])
        else:
            self.vid_neg.write(data["image"])
        
        return None

