import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import sys
import argparse



from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.all import SaveModelCallback, EarlyStoppingCallback
from fastai.data.all import DataBlock, RandomSplitter
from fastai.vision.all import get_image_files, ImageBlock, MaskBlock

# Parser can be used if we are trying to target specific GPUs. 
# If we do this, it might be best to comment out the fastai.distributed import *
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device', type=str, default='None', required=False,
                    help='The cuda index to run on (None for all), (0 for cuda:0), (1 for cuda:1), ...')
args = parser.parse_args()

if args.cuda_device != 'None':
    DEVICE = f'cuda:{args.cuda_device}'
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')


file_dir = os.path.dirname(os.path.realpath("__file__"))
code_root = os.path.join(file_dir, '..')

data_root = Path(code_root) / 'data' # Data root directory
data_subdir = 'starcraft-sensor-dataset'
segment_size = 'segment'

segment_path = data_root / data_subdir / segment_size

n_epochs = 20

# Load StarCraft2Sensor stuff
sys.path.append(str(code_root))  # Needed for import below
from sc2sensor.dataset import StarCraftSensor
from sc2sensor.utils.unit_type_data import NONNEUTRAL_CHANNEL_TO_ID, NONNEUTRAL_ID_TO_NAME
CHANNEL_TO_NAME = [NONNEUTRAL_ID_TO_NAME[NONNEUTRAL_CHANNEL_TO_ID[i]] for i in range(len(NONNEUTRAL_CHANNEL_TO_ID))]

experiments = [
#   ('unet_resnet18', resnet18, 512),
#   ('unet_resnet34', resnet34, 512),
#   ('unet_xresnet18', xresnet18_deep, 512),
#   ('unet_xresnet34', xresnet34_deep, 512),
#    ('unet_squeezenet1_0', squeezenet1_0, 256),
  ('unet_squeezenet1_1', squeezenet1_1, 256),
  ('unet_densenet121', densenet121, 64),
  ('unet_densenet169', densenet169, 64),
]





# doing this in a function to help with memory leaks
def train_model(name, arch, batch_size, learner_kwargs):
        sc2_segment = DataBlock(
                          blocks=(ImageBlock, MaskBlock(codes=CHANNEL_TO_NAME)),
                          get_items=get_image_files,
                          get_y=lambda filename: (os.path.splitext(filename)[0].replace('images','labels') + '_labels.png'),
                          splitter=RandomSplitter(seed=0),
                          batch_tfms=None)
        dls = sc2_segment.dataloaders(segment_path/'train'/'images', shuffle=True, bs=batch_size, num_workers=12)
        callbacks = [
            EarlyStoppingCallback(patience=1),
            SaveModelCallback(fname=name, with_opt=True, every_epoch=True),
            CSVLogger(fname=str(Path(learner_kwargs["model_dir"])/'train_history.csv'))
            ]
        # Create learner
        learner = unet_learner(arch=arch, dls=dls, **learner_kwargs)
        with learner.distrib_ctx():
            learner.fine_tune(n_epochs, cbs=callbacks)
        torch.cuda.empty_cache()
        return None
        
print(f'Starting experiments on segmentation dataset: {str(segment_path)}. ',
      f'Training for {n_epochs} with \n{experiments}')

for name, arch, batch_size in experiments:
    print('\n'*5, f'Starting {name} with size {batch_size}.'.center(60, '-'))
    
    learner_kwargs = dict(path=Path(code_root)/'models'/'unit-type-identification', model_dir=name)
    if not (learner_kwargs['path']/learner_kwargs['model_dir']).exists():
        (learner_kwargs['path']/learner_kwargs['model_dir']).mkdir()

    train_model(name, arch, batch_size, learner_kwargs)
    
    print(f'Finished {name}.')
