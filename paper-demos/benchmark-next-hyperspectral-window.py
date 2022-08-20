from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse

from pathlib import Path
import os, sys

from fastai.data.all import Transform, DataBlock, RandomSplitter, L
#from fastai.vision.all import unet_learner, resnet18, resnet34, xresnet18, xresnet34_deep, xresnet34_deeper
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.all import SaveModelCallback, EarlyStoppingCallback
import segmentation_models_pytorch as smp


# Parser can be used if we are trying to target specific GPUs. 
# If we do this, it might be best to comment out the fastai.distributed import *
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_device', type=str, default='None', required=False,
                    help='The cuda index to run on (None for all), (0 for cuda:0), (1 for cuda:1), ...')
parser.add_argument('--max_samples', type=int, default=None, required=False,
                    help='The max number of samples use from the hyperspectral dataset')
parser.add_argument('--n_epochs', type=int, default=10, required=False,
                    help='The number of training epochs')
args = parser.parse_args()

print(f'Starting with inputs: {str(args)}')

if args.cuda_device != 'None':
    DEVICE = f'cuda:{args.cuda_device}'
    from warnings import warn
    warn('SETTING DEVICE')
    torch.cuda.set_device(DEVICE)
else:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

max_samples = args.max_samples
if max_samples is not None:
    print(f'Using a max number of a {max_samples}')

batch_size = 32


# Load StarCraft2Sensor stuff
ipynb_dir = os.path.dirname(os.path.realpath("__file__"))
model_dir = os.path.join(ipynb_dir, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
code_root = os.path.join(ipynb_dir, '..')
sys.path.append(code_root)  # Needed for import below

from sc2sensor.dataset import StarCraftSensor
from sc2sensor.utils.unit_type_data import NONNEUTRAL_CHANNEL_TO_ID, NONNEUTRAL_ID_TO_NAME
CHANNEL_TO_NAME = [NONNEUTRAL_ID_TO_NAME[NONNEUTRAL_CHANNEL_TO_ID[i]] for i in range(len(NONNEUTRAL_CHANNEL_TO_ID))]


data_root = os.path.join(code_root, 'data') # Data root directory
data_subdir = 'starcraft-sensor-dataset'

# Create subclass of original dataset for next window prediction
class NextWindowDataset(StarCraftSensor):
  def __init__(self, *args, max_samples=None, **kwargs):
    assert 'use_sparse' not in kwargs, 'Cannot set use_sparse with this dataset.'
    assert 'compute_labels' not in kwargs, 'Cannot set use_sparse with this dataset.'
    super().__init__(*args, use_sparse=True, compute_labels=False, **kwargs)
    self.max_samples = max_samples
    
    # Sort data so that next index is merely + 1
    self.metadata = self.metadata.sort_values(['static.replay_name', 'dynamic.window_idx']).reset_index(drop=True)
    md = self.metadata
    
    # Get starting window indices
    start_windows = md[(md['dynamic.num_windows'] > 1) 
                       & (md['dynamic.window_idx'] < (md['dynamic.num_windows'] - 1))]
    self.start_idx = start_windows.index

  def __getitem__(self, idx):
    # Get original indices of start and end based on input 
    md = self.metadata
    # Assumes sorted
    orig_idx = self.start_idx[idx]
    next_idx = orig_idx + 1
    # Only sanity check first and last as this may be expensive
    if idx == 0 or idx == len(self) - 1:
      assert md['static.replay_name'][orig_idx] == md['static.replay_name'][next_idx], 'Replays are not the same'
      assert md['dynamic.window_idx'][orig_idx] + 1 == md['dynamic.window_idx'][next_idx], 'Window indices are not adjacent'

    # Get combined hyperspectral images
    def get_hyperspectral_dense(idx):
      # Concatenate player1 and player2 hyperspectral
      replay_file, window_idx = self._get_replay_and_window_idx(idx)
      with np.load(replay_file) as data:
        player_1_hyperspectral = self._extract_hyperspectral(
          'player_1', data, window_idx)
        player_2_hyperspectral = self._extract_hyperspectral(
          'player_2', data, window_idx)
      return torch.concat([player_1_hyperspectral.to_dense(), 
                           player_2_hyperspectral.to_dense()], 
                          dim=-3).float()
    windows = [get_hyperspectral_dense(idx) for idx in [orig_idx, next_idx]] 
    x = windows[0]
    y = windows[1] - windows[0] # Compute diff
    return x, y

  def __len__(self):
    if self.max_samples is not None:
      return min(self.max_samples, len(self.start_idx))
    else:
      return len(self.start_idx)

# Load datasets
next_window_train = NextWindowDataset(root=data_root, subdir=data_subdir, train=True, max_samples=max_samples)

single = next_window_train[-1]
n_input_channels = single[0].shape[0]  # Needed for preprocessor to get down to 3 channels
print(f'Num train: {len(next_window_train)}') #, Num test: {len(next_window_test)}')
#print(f'Num train: {len(next_window_train)}, Num test: {len(next_window_test)}')
print(f'x.shape: {single[0].shape}, y.shape: {single[1].shape}')

# Create fastai dataloaders given the PyTorch dataset

class AddChannelCodes(Transform):
  "Add the code metadata to a `TensorMask`"
  def __init__(self, codes=None):
      self.codes = codes
      if codes is not None: self.vocab,self.c = codes,len(codes)

  def decodes(self, o):
      if self.codes is not None: o.codes=self.codes
      return o
    
# HACK: Put all instances in both "train" and "valid"
# From https://forums.fast.ai/t/solved-not-splitting-datablock/84759/3
def all_splitter(o): return L(int(i) for i in range(len(o))), L(int(i) for i in range(len(o)))
    
SC2_CODES = [
  f'{player}_{name}'
  for player in ['P1','P2']
  for name in CHANNEL_TO_NAME
]
def create_dataloaders_from_dataset(dataset, splitter=None, **kwargs):
  # Needs to have reference to dataset for closures
  assert 'get_x' not in kwargs
  assert 'get_y' not in kwargs
  assert 'get_items' not in kwargs
  splitter = splitter if splitter is not None else RandomSplitter(seed=0)
  assert len(SC2_CODES) == dataset[0][0].shape[0], 'Number of codes does not match number of channels'
  block = DataBlock(
    get_items=lambda d: list(range(len(d))),
    get_x=lambda idx: dataset[idx][0],
    get_y=lambda idx: dataset[idx][1],
    blocks=None, # These are just transforms
    splitter=splitter,
    item_tfms=[AddChannelCodes(SC2_CODES)],
  )
  return block.dataloaders(dataset, **kwargs)


dls = create_dataloaders_from_dataset(next_window_train, batch_size=batch_size)

experiments = [
  'resnet18',
  'resnet34',
#   'efficientnet-b2',
#   'efficientnet-b4',
#   'densenet121',
#   'densenet169',
]


shared_kwargs = dict(dls=dls, loss_func=nn.MSELoss(), path=Path(code_root))
for arch in experiments:
  name = 'unet_' + arch
  print('\n'*5, f'Starting {name}, with batch size {batch_size}.'.center(60, '-'))

  model_dir = str(Path('models')/'next_window'/name)
  if not (shared_kwargs['path']/model_dir).exists():
    (shared_kwargs['path']/model_dir).mkdir(parents=True)

  model = smp.Unet(
    encoder_name=arch,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=len(SC2_CODES),                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=len(SC2_CODES),                      # model output channels (number of classes in your dataset)
    activation=None, 
  )
  #print(model)
  learner = Learner(model=model, model_dir=model_dir, **shared_kwargs)
  
  callbacks = [
    EarlyStoppingCallback(patience=1),
    SaveModelCallback(fname=f'next_window_{len(next_window_train)}_{name}', with_opt=True, every_epoch=True),
    CSVLogger(fname=str(Path(model_dir)/'train_history.csv'))
  ]
  # Does one epoch of fine tune with frozen weights, then normal epochs unfrozen
  with learner.distrib_ctx():
    learner.fine_tune(args.n_epochs, cbs=callbacks)
  # Now that training has finished, empty the cache
  torch.cuda.empty_cache()

  print(f'Finished {name}.')