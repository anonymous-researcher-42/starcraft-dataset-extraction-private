import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import sys
import argparse
import pandas as pd

from torchvision.io import read_image as torch_read_image

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
parser.add_argument('--n_epochs', type=int, default=10, required=False,
                    help='The number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, required=False,
                    help='The number of batches to use during train/val')

args = parser.parse_args()
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

if segment_size.endswith('small'):
    from warnings import warn
    warn('USING SMALL DEV DATASET')

segment_path = data_root / data_subdir / segment_size

# Load StarCraft2Sensor stuff
sys.path.append(str(code_root))  # Needed for import below
from sc2sensor.dataset import StarCraftSensor
from sc2sensor.utils.unit_type_data import NONNEUTRAL_CHANNEL_TO_ID, NONNEUTRAL_ID_TO_NAME
from sc2sensor.utils.sensor_utils import SensorPlacementDataset, SUPPORTED_PLACEMENT_KINDS
CHANNEL_TO_NAME = [NONNEUTRAL_ID_TO_NAME[NONNEUTRAL_CHANNEL_TO_ID[i]] for i in range(len(NONNEUTRAL_CHANNEL_TO_ID))]

# removing barrier_h and barrier_v from the placement kinds since diag kinds are more fitting
PLACEMENT_KINDS = [kind for kind in SUPPORTED_PLACEMENT_KINDS if not (kind.endswith('_h') or kind.endswith('_v'))]

architectures = [
  ('unet_resnet18', resnet18)
]

experiments = [[(*arch, kind) for arch in architectures] for kind in PLACEMENT_KINDS]  # makes list of experiment lists
experiments = sum(experiments, [])  # join list of lists --> [(arch1, kind1), (arch2, kind1), ..., (arch1, kind2), ...]



# Loading dataset
class SegmentationDataset(torch.utils.data.Dataset):
    
    def __init__(self, segment_path, create_metadata=True):
        super().__init__()
        self.path = Path(segment_path)
        self.X_filenames = self._get_files()
        if create_metadata:
            self.metadata, self.match_metadata = self._make_metadata()
        
    def __len__(self):
        return len(self.X_filenames)
    
    def __getitem__(self, idx):
        X_filename = self.X_filenames[idx]
        y_filename = os.path.splitext(X_filename)[0].replace('images','labels') + '_labels.png'
        
#         return torch_read_image(str(X_filename)), torch_read_image(str(y_filename)).squeeze()
        return (plt.imread(str(X_filename))*255).astype(np.uint8), \
               (plt.imread(str(y_filename))*255).astype(np.uint8)

    def _get_files(self):
        return list((self.path / 'images').glob('**/*.png'))
    
    def _make_metadata(self):
        replay_names = [str(f).split('_')[-2].split('/')[-1] for f in self.X_filenames]
        metadata = pd.DataFrame({'static.replay_name':replay_names})
        match_metadata = metadata.drop_duplicates(subset=['static.replay_name']).reset_index(drop=True)
        return metadata, match_metadata

segmentation_dataset = SegmentationDataset(segment_path/'train', create_metadata=True)

# doing this in a function to help with memory leaks
def train_model(name, arch, kind, segmentation_dataset, learner_kwargs):

    # Creating sensor dataset inside function to stop memory leaks
    # (I think it's from using the same segmentation dataset in multiple datablocks, but I'm not sure)
    # creating the datablocks/loaders in this function seems to work though.
    segmentation_sensor_placement_dataset = SensorPlacementDataset(
            segmentation_dataset, n_sensors=50, radius=5.5, 
            kind=kind, failure_rate=0.2, return_mask=False, make_cache=False)

    block = DataBlock(
        blocks=(ImageBlock, MaskBlock),
        get_items=lambda d: list(range(len(d))),
        get_x=lambda idx: segmentation_sensor_placement_dataset[idx][0],
        get_y=lambda idx: segmentation_sensor_placement_dataset[idx][1],
        splitter=RandomSplitter(seed=0),
    )
    dls = block.dataloaders(segmentation_sensor_placement_dataset, batch_size=args.batch_size)

    callbacks = [
        EarlyStoppingCallback(patience=1),
        SaveModelCallback(fname=name, with_opt=True, every_epoch=True),
        CSVLogger(fname=str(Path(learner_kwargs["model_dir"])/'train_history.csv'))
        ]
    # Create learner
    learner = unet_learner(arch=arch, dls=dls, **learner_kwargs)
    with learner.distrib_ctx():
        learner.fine_tune(args.n_epochs, cbs=callbacks)
    torch.cuda.empty_cache()
    return None
        
print(f'Starting experiments on segmentation dataset: {str(segment_path)}. ',
      f'Training for {args.n_epochs} epochs with \n{experiments}')

for name, arch, kind in experiments:
    print('\n'*5, f'Starting {name} with kind {kind}.'.center(60, '-'))
    
    learner_kwargs = dict(path=Path(code_root)/ 'models' / 'sensor_unit_identification' / f'{kind}_sensors',
                          model_dir=name,
                          n_out=170  # WARNING: I'm not 100% sure if this is correct, but I think it is.
                          # TODO: check if setting n_out=170 is correct.
                          )
    if not (learner_kwargs['path']/learner_kwargs['model_dir']).exists():
        # first check if main model dir exists
        if not learner_kwargs['path'].exists():
            learner_kwargs['path'].mkdir(parents=True)
        # then check if sub model dir exists
        (learner_kwargs['path']/learner_kwargs['model_dir']).mkdir()

    train_model(name, arch, kind, segmentation_dataset, learner_kwargs)
    
    print(f'Finished {name} with kind {kind}.')
