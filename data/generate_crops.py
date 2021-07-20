import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from argparse import ArgumentParser

from .utils import get_video_paths


def convert_to_square(bbox):
    bbox = np.array(bbox)
    square_bbox = bbox.copy()

    h = bbox[3] - bbox[1] + 1
    w = bbox[2] - bbox[0] + 1
    max_side = np.maximum(h, w)
    square_bbox[0] = bbox[0] + w * 0.5 - max_side * 0.5
    square_bbox[1] = bbox[1] + h * 0.5 - max_side * 0.5
    square_bbox[2] = square_bbox[0] + max_side - 1
    square_bbox[3] = square_bbox[1] + max_side - 1
    return square_bbox


def crop_x2_replicate_300(img, coords):
    x, y, x1, y1 = coords
    dh, dw, ch = img.shape

    w = x1 - x
    h = y1 - y
    wb = w // 2
    hb = h // 2
        
    top_bord = min(0, y - hb) * -1
    left_bord = min(0, x - wb) * -1    
    bottom_bord = min(0, dh - (y1 + hb)) * -1
    right_bord = min(0, dw - (x1 + wb)) * -1

    crop = img[max(0, y - hb):min(dh, y1 + hb), max(0, x - wb):min(dw, x1 + wb)]
    border_crop = cv2.copyMakeBorder(crop, 
                                     top=top_bord,
                                     bottom=bottom_bord,
                                     left=left_bord,
                                     right=right_bord,
                                     borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(border_crop,(300,300))


def extract_crops(video_path, 
                  bboxes_folder, 
                  crops_folder, 
                  root_dir):
    bboxes_dir = os.path.join(root_dir, bboxes_folder)
    sub_path = video_path.split(root_dir)[-1]
    whole_id = os.path.splitext(sub_path)[0].replace('/', '@')
    bboxes_path = os.path.join(bboxes_dir, f"{whole_id}.json")
    with open(bboxes_path, 'r') as f:
        bboxes_dict = json.load(f)
        
    capture = cv2.VideoCapture(video_path)
    frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frames_num):
        capture.grab()
        if str(i) not in bboxes_dict:
            continue
        
        success, frame = capture.retrieve()
        if not success:
            continue
        
        bbox = bboxes_dict[str(i)]
        if bbox is None:
            continue
        
        bbox = convert_to_square(bbox)
#         bbox = [int(b * 2) for b in bbox] # because of the 0.5x downscaling during face extraction phase
        bbox = [int(b) for b in bbox]
        
        crop = crop_x2_replicate_300(frame, bbox)
        
        crops_out_dir = os.path.join(root_dir, crops_folder, whole_id)
        os.makedirs(crops_out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(crops_out_dir, f"{i}.jpg"), crop) # it's better to save as jpg
        
        
if __name__ == '__main__':
    parser = ArgumentParser(description='Extract crops from original videos using pre-defined bboxes')
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--bboxes-folder", default=None)
    parser.add_argument("--crops-folder", default=None)
    cmd_args = parser.parse_args()
    print("Command line args:", cmd_args)
    videos_paths = get_video_paths(cmd_args.root_dir)
    for video in videos_paths:
        extract_prod_crops(video, 
                           bboxes_folder=cmd_args.bboxes_folder,
                           crops_folder=cmd_args.crops_folder,
                           root_dir=cmd_args.root_dir)            
    