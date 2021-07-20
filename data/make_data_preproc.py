import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import multiprocessing
from joblib import Parallel, delayed
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from .generate_bboxes import extract_bboxes
from .generate_crops import extract_crops
from .mp_patch import patch_mp_connection_bpo_17560


VIDEO_TYPES = ['.3GP', '.3gp', '.avi', '.AVI', '.mov', '.MOV', '.MP4', '.mp4', '.webm']


def get_video_paths(root_dir, extensions=['.mp4', '.avi']):
    paths = []
    for path in Path(root_dir).rglob('*.*'):
        if path.suffix in extensions:
            paths.append(str(path))
            
    return paths    


def main(root_dir, bboxes_folder, crops_folder):

    ready_data = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    
    for data_folder in tqdm(ready_data, dynamic_ncols=True, desc=f'working with folders'):
        videos = get_video_paths(root_dir=data_folder, extensions=VIDEO_TYPES)
        if len(videos) == 0:
            continue
        
        extract_bboxes(video_paths=videos, frame_divisor=1, size_divisor=1,
                       root_dir=root_dir, bboxes_folder=bboxes_folder)
        
        num_cores = multiprocessing.cpu_count() // 4
        
        Parallel(n_jobs=num_cores)(delayed(extract_crops)(video, bboxes_folder, crops_folder, \
                                                          root_dir) for video in tqdm(videos, 'extracting crops'))


if __name__ == '__main__':
    parser = ArgumentParser(description='Extract crops from original videos using pre-defined bboxes')
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--bboxes-folder", default=None)
    parser.add_argument("--crops-folder", default=None)
    cmd_args = parser.parse_args()
    print("Command line args:", cmd_args)
    patch_mp_connection_bpo_17560()
    main(root_dir=cmd_args.root_dir,
        bboxes_folder=cmd_args.bboxes_folder,
        crops_folder=cmd_args.crops_folder)

