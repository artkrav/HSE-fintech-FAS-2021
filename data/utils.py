import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import copy
import random
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict
from PIL import Image
                                

def get_video_frames(video_path, frame_divisor=10, size_divisor=2, output_format='pil'):
    assert output_format in {'pil', 'cv2'}, 'inappropriate output format'
    assert size_divisor != 0, 'frame size division by zero'
    capture = cv2.VideoCapture(video_path)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = OrderedDict()
    for i in range(frames_num):
        capture.grab()
        if i % frame_divisor != 0:
            continue
        
        flag, frame = capture.retrieve()
        if not flag:
            continue
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        down_size = [s // size_divisor for s in frame.shape[:2]] # downsize frame to size divided by `size_divisor` of initial shape
        h, w = down_size
        if output_format == 'pil':
            frame = Image.fromarray(frame)
            frame = frame.resize(size=(w, h))
        else:
            frame = cv2.resize(frame, (w, h))
        frames[i] = frame
    capture.release()
    return frames   


class VideoDataset(Dataset):    
    def __init__(self, video_paths, frame_divisor, size_divisor, output_format='pil'):
        super().__init__()
        self.videos = video_paths
        self.frame_divisor = frame_divisor
        self.size_divisor = size_divisor
        self.output_format  = output_format
        
    def __getitem__(self, index):
        video = self.videos[index]
        frames = get_video_frames(video, 
                                  frame_divisor=self.frame_divisor,
                                  size_divisor=self.size_divisor,
                                  output_format=self.output_format)
        return video, list(frames.keys()), list(frames.values())
    
    def __len__(self):
        return len(self.videos) 
    

def get_video_paths(root_dir, extensions=['.mp4', '.avi']):
    paths = []
    for path in Path(root_dir).rglob('*.*'):
        if path.suffix in extensions:
            paths.append(path)
            
    return paths    
    