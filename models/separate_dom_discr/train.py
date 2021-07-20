import os
import random
import torch
import time
import numpy as np
import torch.nn.functional as F
import sklearn.metrics as metrics
from pydoc import locate
from pathlib import Path
from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm 

from models.dataloader import (LivenessDataset, 
                               prepare_dataloaders, 
                               prepare_datasets,
                               get_weights_for_sampling)
from models.zoo.domain_discriminators import DAFL_FAS
from training.config_parser import read_config
from training.metric import AverageMeter, Tracker
from training.experiment import Experiment


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def to_device(*args, device):
    return [a.to(device, non_blocking=True) for a in args]


def build_model(args):
    model = DAFL_FAS(
        encoder=args.encoder_type,
        dropout_rate=args.dropout_rate,
        revgrad_lamba=args.revgrad_lambda,
        num_domains=args.num_domains,
        single_discriminator=False
    )
    return model


def build_loss_func(loss_type):
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    def loss_func(input, target):
        if loss_type == 'cross_entropy':
            return cross_entropy(input, target.long())
        elif loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(input, target, reduction='mean')
        else:
            raise ValueError(loss_type)
    return loss_func


def build_scheduler(args):
    def scheduler_(optimizer):
        if args.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, **args.scheduler_params)
        elif args.scheduler == 'multistep':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **args.scheduler_params)
        elif args.scheduler == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **args.scheduler_params)
        elif args.scheduler == "constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        else:
            raise ValueError(args.scheduler)
    return scheduler_
    
    
def train(args: Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    ex = Experiment(args)
    
    train_dataset, dev_dataset, test_dataset = prepare_datasets(args)
    train_dataset.reset(epoch=0, seed=args.seed)
    train_loader, dev_loader, test_loader = prepare_dataloaders(args, 0, train_dataset, dev_dataset, test_dataset)
    
    device = torch.device('cuda:0')
    model = build_model(args).to(device)
    optimizer = locate(args.optimizer_class)(model.parameters(), **args.optimizer_hyperparams)
    scheduler = build_scheduler(args)(optimizer)
    
    loss_func_cls = build_loss_func(args.cls_loss_type)
    loss_func_ccdd = build_loss_func(args.ccdd_loss_type)
        
    train_loss = AverageMeter()
    train_live_loss = AverageMeter()
    train_spoof_loss = AverageMeter()
    train_dis_live_loss = AverageMeter()
    train_dis_spoof_loss = AverageMeter()
    
    dev_loss = AverageMeter()
    dev_live_loss = AverageMeter()
    dev_spoof_loss = AverageMeter()
    
    test_loss = AverageMeter()
    test_live_loss = AverageMeter()
    test_spoof_loss = AverageMeter()
    
    for epoch in range(0, args.epochs):
        if epoch != 0:
            train_dataset.reset(epoch=epoch, seed=args.seed)
            train_loader, dev_loader, test_loader = prepare_dataloaders(args, epoch, train_dataset, dev_dataset, test_dataset)
    
        train_loss.reset()
        train_live_loss.reset()
        train_spoof_loss.reset()
        train_dis_live_loss.reset()
        train_dis_spoof_loss.reset()
        
        dev_loss.reset()
        dev_live_loss.reset()
        dev_spoof_loss.reset()
        
        test_loss.reset()
        test_live_loss.reset()
        test_spoof_loss.reset()
        
        ex.log.info(f'Epoch {epoch} learning rate {optimizer.param_groups[0]["lr"]:.7f}')
        
        model.train()
        for (data, label, domain_label, multi_label) in tqdm(train_loader, dynamic_ncols=True, desc=f"epoch {epoch} train"):
            optimizer.zero_grad()
            data, label, domain_label, multi_label = to_device(data, label, domain_label, multi_label, device=device)
            
            logits_cls, logits_dis_live, logits_dis_spoof = model(data)
            logits_cls = logits_cls.view(-1)
            
            live_loss = 0.
            spoof_loss = 0.
            live_dis_loss = 0.
            spoof_dis_loss = 0.
            
            live_idx = label > 0.5
            spoof_idx = label <= 0.5
            
            if torch.sum(live_idx * 1) > 0:
                live_loss = loss_func_cls(logits_cls[[live_idx]], label[[live_idx]])
                live_dis_labels = domain_label[[live_idx]]
                live_dis_logits = logits_dis_live[[live_idx]]
                live_dis_loss = loss_func_ccdd(live_dis_logits, live_dis_labels)
            
            if torch.sum(spoof_idx * 1) > 0:
                spoof_loss = loss_func_cls(logits_cls[[spoof_idx]], label[[spoof_idx]])
                spoof_dis_labels = domain_label[[spoof_idx]]
                spoof_dis_logits = logits_dis_spoof[[spoof_idx]]
                spoof_dis_loss = loss_func_ccdd(spoof_dis_logits, spoof_dis_labels)
            
            train_live_loss.append(0. if live_loss == 0. else live_loss.detach().item())
            train_spoof_loss.append(0. if spoof_loss == 0. else spoof_loss.detach().item())
            train_dis_live_loss.append(0. if live_dis_loss == 0. else live_dis_loss.detach().item())
            train_dis_spoof_loss.append(0. if spoof_dis_loss == 0. else spoof_dis_loss.detach().item())
            
            loss = loss_func_cls(logits_cls, label) + args.ccdd_lambda * (live_dis_loss + spoof_dis_loss)
            loss.backward()
            train_loss.append(loss.detach().item())
            
            optimizer.step()
            scheduler.step()
           
        tracker_dev = Tracker()
        tracker_test = Tracker()
        model.eval()
        with torch.no_grad(): 
            for (data, label, domain_label, multi_label) in tqdm(dev_loader, dynamic_ncols=True, desc=f"epoch {epoch} dev"):
                data, label, domain_label, multi_label = to_device(data, label, domain_label, multi_label, device=device)
                
                logits_cls, logits_dis_live, logits_dis_spoof = model(data)
                logits_cls = logits_cls.view(-1)
                
                live_loss = 0.
                spoof_loss = 0.
                
                live_idx = label > 0.5
                spoof_idx = label <= 0.5
                
                if torch.sum(live_idx * 1) > 0:
                    live_loss = loss_func_cls(logits_cls[[live_idx]], label[[live_idx]])
                if torch.sum(spoof_idx * 1) > 0:
                    spoof_loss = loss_func_cls(logits_cls[[spoof_idx]], label[[spoof_idx]])
                
                dev_live_loss.append(0. if live_loss == 0. else live_loss.detach().item())
                dev_spoof_loss.append(0. if spoof_loss == 0. else spoof_loss.detach().item())
                
                loss = loss_func_cls(logits_cls, label)
                dev_loss.append(loss.detach().item())
                
                pred = logits_cls.sigmoid().detach()  
                tracker_dev.append_batch(scores=pred, targets=label)
                                
            for (data, label, domain_label, multi_label) in tqdm(test_loader, dynamic_ncols=True, desc=f"epoch {epoch} test"):
                data, label, domain_label, multi_label = to_device(data, label, domain_label, multi_label, device=device)
                
                logits_cls, logits_dis_live, logits_dis_spoof = model(data)
                logits_cls = logits_cls.view(-1)
                
                live_loss = 0.
                spoof_loss = 0.
                
                live_idx = label > 0.5
                spoof_idx = label <= 0.5
                
                if torch.sum(live_idx * 1) > 0:
                    live_loss = loss_func_cls(logits_cls[[live_idx]], label[[live_idx]])
                if torch.sum(spoof_idx * 1) > 0:
                    spoof_loss = loss_func_cls(logits_cls[[spoof_idx]], label[[spoof_idx]])
                
                test_live_loss.append(0. if live_loss == 0. else live_loss.detach().item())
                test_spoof_loss.append(0. if spoof_loss == 0. else spoof_loss.detach().item())
                
                loss = loss_func_cls(logits_cls, label)
                test_loss.append(loss.detach().item())
                
                pred = logits_cls.sigmoid().detach()  
                tracker_test.append_batch(scores=pred, targets=label)
                
        dev_metrics = {
            'auprc_dev': tracker_dev.auprc(), 
            'auroc_dev': tracker_dev.auroc(),
            'eer_dev': tracker_dev.eer()[0], 
            'eer_thr_dev': tracker_dev.eer()[1],
        }
        
        test_metrics = {
            'auprc_test': tracker_test.auprc(),
            'auroc_test': tracker_test.auroc(),
            'eer_test': tracker_test.eer()[0], 
            'eer_thr_test': tracker_test.eer()[1],
            'hter_test': tracker_test.hter(tracker_dev.eer()[1])[0],
            'hter_thr_test': tracker_test.hter(tracker_dev.eer()[1])[1]
        }
        
        train_losses = {
            'train_loss': train_loss.value,
            'train_live_loss': train_live_loss.value,
            'train_spoof_loss': train_spoof_loss.value,
            'train_dis_live_loss': train_dis_live_loss.value,
            'train_dis_spoof_loss': train_dis_spoof_loss.value
        }
        
        dev_losses = {
            'dev_loss': dev_loss.value,
            'dev_live_loss': dev_live_loss.value,
            'dev_spoof_loss': dev_spoof_loss.value
        }
        
        test_losses = {
            'test_loss': test_loss.value,
            'test_live_loss': test_live_loss.value,
            'test_spoof_loss': test_spoof_loss.value
        }
        
        ex.log_scalars(train_losses, epoch)
        ex.log_scalars(dev_losses, epoch)
        ex.log_scalars(test_losses, epoch)        
        
        ex.log_scalars(dev_metrics, epoch)
        ex.log_scalars(test_metrics, epoch)
        ex.log.info(f'Epoch {epoch} train complete')
        
        ex.checkpoint(model=model, score=1-tracker_dev.eer()[0])
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', default='config.yaml')
    cmd_args = parser.parse_args()
    config = read_config(Path(__file__).parent / cmd_args.cfg)
    train(config)