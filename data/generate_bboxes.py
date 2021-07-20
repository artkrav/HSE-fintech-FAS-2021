import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data.dataloader import DataLoader
from facenet_pytorch.models.mtcnn import MTCNN

from .utils import VideoDataset, get_video_paths


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(margin=0,thresholds=[0.85, 0.95, 0.95], device=device) # higher thresholds for less FP predicts


def extract_bboxes(video_paths, 
                   frame_divisor=10, 
                   size_divisor=2, 
                   root_dir='/Users', 
                   bboxes_folder='boxes'):
    dataset = VideoDataset(video_paths, frame_divisor=frame_divisor, size_divisor=size_divisor, output_format='pil')
    loader = DataLoader(dataset, shuffle=False, num_workers=10, batch_size=1, collate_fn=lambda x: x)
    detector_bs = 16
    for video in tqdm(loader, dynamic_ncols=True, desc='extracting bboxes'):
        result = {}
        video_name, indices, frames = video[0]
        batches = [frames[i: i + detector_bs] for i in range(0, len(frames), detector_bs)]
        batches_ixs = [indices[i: i + detector_bs] for i in range(0, len(indices), detector_bs)]
        for b_frames, b_indices in zip(batches, batches_ixs):
            batch_boxes, *_ = detector.detect(b_frames, landmarks=False)
            batch_boxes = [b[0].tolist() if b is not None else None for b in batch_boxes]
            result.update({i: b for i, b in zip(b_indices, batch_boxes)})
        sub_path = video_name.split(root_dir)[-1]
        id = os.path.splitext(sub_path)[0].replace('/', '@')
        out_dir = os.path.join(root_dir, bboxes_folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{id}.json"), "w") as f:
            json.dump(result, f)
             
            
if __name__ == '__main__':
    parser = ArgumentParser(description='Extract bboxes from original videos by MTCNN face detector')
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--bboxes-folder", default=None)
    parser.add_argument("--frame-divisor", default=10)
    parser.add_argument("--size-divisor", default=2)
    cmd_args = parser.parse_args()
    print("Command line args:", cmd_args)
    videos_paths = get_video_paths(cmd_args.root_dir)
    extract_bboxes(videos_paths, 
                   frame_divisor=cmd_args.frame_divisor,
                   size_divisor=cmd_args.size_divisor,
                   root_dir=cmd_args.root_dir, 
                   bboxes_folder=cmd_args.bboxes_folder)
            