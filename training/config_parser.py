import os
import yaml
import pathlib
import argparse


def check_keys(config):
    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(k, str):
                if '.' in k:
                    raise ValueError(f'Inapproptiate symbol . in config: key {k}')
            check_keys(v)
        

def read_config(path):
    config_path = pathlib.Path(path)
    if not config_path.exists():
        raise FileExistsError('Path {} does not exist'.format(path))
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.loader.FullLoader)
    check_keys(config)
    return argparse.Namespace(**config)
        

