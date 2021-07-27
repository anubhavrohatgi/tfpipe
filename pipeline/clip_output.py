import os
import cv2
import json

from tfpipe.core.utils import save_video
from tfpipe.pipeline.pipeline import Pipeline

class ClipOutput(Pipeline):
    """ Pipeline task the outputs each frame from a clip to seperate videos 
    depending on the detection of an object in the frame."""

    def __init__(self, dok, output_dir, meta, segment, clip_dur, detect_percent, video_ext=".mp4"):
        self.dok = dok
        self.output_dir = output_dir
        self.meta = meta
        self.metadata = dict()
        self.segment = segment
        self.clip_dur = clip_dur
        self.video_ext = video_ext

        self.fps = self.resolution = None

        ## Segment
        self.cache = None
        self.clip_name = self.pos_output_dir = self.neg_output_dir = None
        self.dtct_pct = detect_percent
        
        ## Other
        self.vid_pos = self.vid_neg = None

        super().__init__()

    def map(self, data):
        if self.segment:
            self.segment_output(data)
        else:
            if data[self.dok]:
                self.vid_pos.write(data["image"])
            else:
                self.vid_neg.write(data["image"])
            
            if self.meta:
                self.metadata[data["image_id"]] = data[self.dok]
            
        
        return None

    def segment_output(self, data):
        self.cache.append(data)

        if len(self.cache) == self.MAX_SIZE:
            self.write_cache()
            
    def write_cache(self):
        if len(self.cache) > 0:
            fs, fe = self.cache[0]["image_id"], self.cache[-1]["image_id"]
            num_pos = sum([1 for data in self.cache if data[self.dok]])

            is_pos = num_pos / self.MAX_SIZE >= self.dtct_pct

            out_vid = f"{self.clip_name}_{fs}-{fe}{self.video_ext}"
            out_path = os.path.join(self.pos_output_dir if is_pos else self.neg_output_dir, out_vid)

            frames = [data["image"] for data in self.cache]
            save_video(frames, self.fps, self.resolution, out_path)

            if self.meta:
                for data in self.cache:
                    self.metadata[data["image_id"]] = is_pos

            self.cache.clear()



    def prep_output(self, path, fps, resolution):
        self.clip_name = os.path.splitext(os.path.basename(path))[0]

        if self.segment:
            self.MAX_SIZE = int(self.clip_dur * fps)
            self.cache = list()

            self.pos_output_dir = os.path.join(self.output_dir, self.clip_name + "-pos")
            self.neg_output_dir = os.path.join(self.output_dir, self.clip_name + "-neg")
            os.makedirs(self.pos_output_dir, exist_ok=True)
            os.makedirs(self.neg_output_dir, exist_ok=True)
        else:
            pos_output = os.path.join(self.output_dir, f"{self.clip_name}-pos{self.video_ext}")
            neg_output = os.path.join(self.output_dir, f"{self.clip_name}-neg{self.video_ext}")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.vid_pos = cv2.VideoWriter(pos_output, fourcc, fps, resolution)
            self.vid_neg = cv2.VideoWriter(neg_output, fourcc, fps, resolution)
        
        if self.meta:
            pass

        self.fps = fps
        self.resolution = resolution

    def write_clips(self):
        if self.segment:
            self.write_cache()
        else:
            self.vid_pos.release()
            self.vid_neg.release()
        
        if self.meta:
            meta_path = os.path.join(self.output_dir, f"{self.clip_name}-metadata.json")
            print(f"Writing metadata to: {meta_path}")
            with open(meta_path, "w") as f:
                json.dump(self.metadata, f)


       
    