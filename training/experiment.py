import yaml
import mlflow 
import random 
from argparse import Namespace
from pathlib import Path

from .logger import get_logger
from .checkpoint import ModelCheckpoint


log = get_logger(__name__)


def unpack_dict(d: dict, sep: str = '/', prefix='') -> dict:
    """
    Args:
        d (dict): input nested dict
        sep (str, optional): separator for nested key name. Defaults to '/'.
        prefix (str, optional): prefix for nested key name. Defaults to ''.

    Returns:
        dict: flat dict
    """
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for sk, sv in unpack_dict(v, sep=sep, prefix=k).items():
                if prefix == '':
                    key = sk
                else:
                    key = prefix + sep + str(sk)
                r[key] = sv
        else:
            if prefix == '':
                key = k
            else:
                key = prefix + sep + str(k)
            r[key] = v
    return r


class Experiment:
    """
    Base class for experiments tracking
    """
    def __init__(self, args: Namespace):
        self.args = args
        self.log = get_logger(args.run_name)
        
        self.experiment_dir = (Path(self.args.model_dir) / self.args.experiment_name / self.args.run_name).resolve()
        if self.experiment_dir.exists():
            raise ValueError(f"Experiment dir {self.experiment_dir} already exists")
        
        self.experiment_dir.mkdir(parents=True)
        self.log.info(f'experiment dir: {self.experiment_dir}')
        
        self.checkpoints_dir = self.experiment_dir / 'checkpoints'
        self.config_filename = self.experiment_dir / 'config.yaml'
        with self.config_filename.open('w') as f:
            yaml.dump(vars(self.args), f)
            self.log.info(f'config saved to {self.config_filename}')
            
        self.checkpointer = ModelCheckpoint(str(self.checkpoints_dir), 'checkpoint', 
                                                n_saved=self.args.keep_checkpoints)
        
        mlflow.set_tracking_uri(self.args.mlflow_server)
        mlflow.set_experiment(self.args.experiment_name)
        mlflow.start_run(run_name=self.args.run_name)
        
        run_params = {}
        for k, v in unpack_dict(vars(self.args), sep='.').items():
            run_params[k] = str(v)
            self.log.info(f'param: {k} = {v}')
        mlflow.log_params(run_params)
        
    def log_scalar(self, key, value, step=None, write_log=True):
        if self.args.use_mlflow:
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                print(f'MLFlow error: {e}')
        if write_log:
            self.log.info(f'Step {step} metric {key} = {value:.5f}')
            
    def log_scalars(self, scalars, step=None, write_log=True):
        if self.args.use_mlflow:
            try:
                mlflow.log_metrics(scalars, step=step)
            except Exception as e:
                print(f'MLFlow error: {e}')
        if write_log:
            for k,v in scalars.items():
                self.log.info(f'Step {step} metric {k} = {v:.5f}')
                
    def checkpoint(self, model, score):
        self.checkpointer({'model': model}, score=score)
                