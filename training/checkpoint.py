import os
import tempfile
import torch


class ModelCheckpoint:
    def __init__(self, dirname, filename_prefix,
                 n_saved=1, atomic=True,
                 require_empty=True, create_dir=True):
        """Args:
            dirname (str): directory to save checkpoints
            filename_prefix (str): prefix for the filenames
            n_saved (int, optional): Maximum number of saved checkpoints. Defaults to 1.
            atomic (bool, optional): if True, objects are serialized to a temporary file
            and then moved to final destination in order to avoid damaging. Defaults to True.
            require_empty (bool, optional): if True, raises exception if there are any files
            starting with 'filename_prefix'. Defaults to True.
            create_dir (bool, optional): creates directory 'dirname' if it doesn't exist. Defaults to True.
        """
        self._dirname = os.path.expanduser(dirname)
        self._fname_prefix = filename_prefix
        self._n_saved = n_saved
        self._atomic = atomic
        self._saved = [] 
        self._iteration = 0
        
        if create_dir:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                
        if not os.path.exists(dirname):
            raise ValueError(f'Directory path {dirname} is not found')
        
        if require_empty:
            matched = [fn for fn in os.listdir(dirname) if fn.startswith(self._fname_prefix)]
            if len(matched) > 0:
                raise ValueError(f'Files prefixed with {filename_prefix} are already present '
                                 f'in the directory {dirname}. If you want to use this '
                                 'directory anyway, pass `require_empty=False`')
                
    def _save(self, obj, path):
        if not self._atomic:
            self._internal_save(obj, path)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self._dirname)
            try:
                self._internal_save(obj, tmp.file)
            except BaseException:
                tmp.close()
                os.remove(tmp.name)
                raise
            else:
                tmp.close()
                os.rename(tmp.name, path)
                print(f'Wrote checkpoint to {path}')
                
    def _internal_save(self, obj, path):
        if hasattr(obj, 'state_dict'):
            torch.save(obj.state_dict(), path)
        else:
            torch.save(obj, path)
            
    def __call__(self, to_save, score):
        if len(to_save) == 0:
            raise RuntimeError('No objects to checkpoint found')
        
        self._iteration += 1
        
        priority = score
        
        for name, obj in to_save.items():
            fname = f'{self._fname_prefix}_{name}_latest.pth'
            path = os.path.join(self._dirname, fname)
            self._save(obj=obj, path=path)
            
        if (len(self._saved) < self._n_saved) or (self._saved[0][0] < priority):
            saved_objs = []
            
            for name, obj in to_save.items():
                fname = f'{self._fname_prefix}_{name}_{self._iteration}_score_{abs(priority):.7f}.pth'
                path = os.path.join(self._dirname, fname)
                
                self._save(obj=obj, path=path)
                saved_objs.append(path)
                
            self._saved.append((priority, saved_objs))
            self._saved.sort(key=lambda item: item[0])
            
        if len(self._saved) > self._n_saved:
            _, paths = self._saved.pop(0)
            for p in paths:
                os.remove(p)
            