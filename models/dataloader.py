import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
import albumentations as A

from torch.utils.data import Dataset, DataLoader


multi_label_map = {
    'print': 0,
    'live': 1,
    'replay': 2
}


def image_transform(image_size, crop_size):
    h_resize, w_resize = image_size
    h_crop, w_crop = crop_size
    return A.Compose([
        A.Resize(height=h_resize, width=w_resize, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.CenterCrop(height=h_crop, width=w_crop, p=1.0),
        A.Normalize(p=1.0)
    ])
    

class LivenessDataset(Dataset):
    def __init__(self, 
                 root_dir: str,
                 crops_folder: str,
                 split_path: str,
                 eqbinary_sample_data: bool,
                 eqmulti_sample_data: bool,
                 image_size=(300, 300),
                 crop_size=(224, 224),
                 is_dev=False,
                 ):
        super().__init__()
        if all([eqbinary_sample_data, eqmulti_sample_data]):
            raise ValueError('Choose only one sampling strategy')
        self.root_dir = root_dir
        self.crops_folder = crops_folder
        self.split_path = split_path
        self.eqbinary_sample_data = eqbinary_sample_data
        self.eqmulti_sample_data = eqmulti_sample_data
        self.image_size = image_size
        self.crop_size = crop_size
        self.is_dev = is_dev
        
        self.split = pd.read_csv(split_path, index_col=0)
        if self.is_dev:
            self.data = pd.read_csv(split_path, index_col=0)
        
    def __getitem__(self, index: int):
        item = self.data.iloc[index]
        frame_path = os.path.join(self.root_dir, self.crops_folder, item.frame_path)
        label = np.array(item.label).astype(np.float32)
        domain_label = np.array(item.domain_label).astype(np.float32)
        multi_label = np.array(multi_label_map[item.multiclass_label]).astype(np.float32)
        
        img = cv2.imread(frame_path, cv2.IMREAD_COLOR)[...,::-1]
        transform = image_transform(self.image_size, self.crop_size)
        img = transform(image=img)['image'].transpose(2, 0, 1).astype(np.float32)
        return img, label, domain_label, multi_label
    
    def reset(self, epoch: int, seed: int):
        self.data = self._prepare_data(epoch, seed)
        
    def _prepare_data(self, epoch: int, seed: int):
        split = self.split.copy()
        seed = (epoch + 1) * seed
        if self.eqbinary_sample_data:
            data = self._eqbinary_manual_sample(split, seed)
        elif self.eqmulti_sample_data:
            data = self._eqmulti_manual_sample(split, seed)    
        else:
            data = split
        return data.sample(frac=1, random_state=seed)
    
    def _eqbinary_manual_sample(self, split, seed, n_dom=3):
        agg = split.groupby(['domain_label', 'label'], as_index=False)[['frame_path']].count()
        num_min = agg.frame_path.min()
        conc = []
        for dl in range(n_dom):
            for bl in range(2):
                sampled = split[(split.label == bl) 
                                & (split.domain_label == dl)].sample(n=num_min, replace=False, random_state=seed)
                conc.append(sampled)
        conc = pd.concat(conc, ignore_index=True)
        return conc
    
    def _eqmulti_manual_sample(self, split, seed, n_dom=3):
        agg = split.groupby(['domain_label', 'multiclass_label'], as_index=False)[['frame_path']].count()
        num_min = agg.frame_path.min()
        conc = []
        for dl in range(n_dom):
            for ml in multi_label_map.keys():
                sampled = split[(split.multiclass_label == ml) 
                                & (split.domain_label == dl)].sample(n=num_min, replace=False, random_state=seed)
                conc.append(sampled)
        conc = pd.concat(conc, ignore_index=True)
        return conc
    
    def __len__(self):
        return len(self.data)
    
    
def get_weights_for_sampling(labels):
    indicies = range(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    mapping = {k: v for k, v in zip(classes, counts)}
    weights = [1.0 / mapping[labels[idx]] for idx in indicies]
    return torch.DoubleTensor(weights)    

    
def prepare_datasets(args):
    train_dataset = LivenessDataset(root_dir=args.root_dir,
                                    crops_folder=args.crops_folder,
                                    split_path=args.train_split,
                                    eqbinary_sample_data=args.eqbinary_sample_data,
                                    eqmulti_sample_data=args.eqmulti_sample_data,
                                    image_size=args.image_size,
                                    crop_size=args.crop_size,
                                    is_dev=False)
    
    dev_dataset = LivenessDataset(root_dir=args.root_dir,
                                  crops_folder=args.crops_folder,
                                  split_path=args.dev_split,
                                  eqbinary_sample_data=args.eqbinary_sample_data,
                                  eqmulti_sample_data=args.eqmulti_sample_data,
                                  image_size=args.image_size,
                                  crop_size=args.crop_size,
                                  is_dev=True)
    
    test_dataset = LivenessDataset(root_dir=args.root_dir,
                                   crops_folder=args.crops_folder,
                                   split_path=args.test_split,
                                   eqbinary_sample_data=args.eqbinary_sample_data,
                                   eqmulti_sample_data=args.eqmulti_sample_data,
                                   image_size=args.image_size,
                                   crop_size=args.crop_size,
                                   is_dev=True)
    return train_dataset, dev_dataset, test_dataset


def prepare_dataloaders(args, epoch, train_dataset, dev_dataset, test_dataset):    
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
        
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)
        
    val_loader = DataLoader(test_dataset, batch_size=args.test_batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)
        
    return train_loader, dev_loader, val_loader
