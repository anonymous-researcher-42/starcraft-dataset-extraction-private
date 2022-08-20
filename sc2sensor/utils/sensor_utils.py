import torch
import numpy as np
from torch.utils.data import Dataset

SUPPORTED_PLACEMENT_KINDS = ['grid', 'random', 'quasi_random',
                             'barrier_h', 'barrier_v', 'barrier_d1', 'barrier_d2']


class Sensor():
  def __init__(self, location, radius, input_shape):
    self.location = location
    self.radius = radius
    self.input_shape = input_shape
    
  def get_mask(self):

    h, w = self.input_shape
    center = self.location
    radius = self.radius

    Y, X = torch.arange(h).reshape(-1,1), torch.arange(w).reshape(1,-1)
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
  # Legacy, keeping just in case
  def OLD_get_mask(self):
    mask = torch.zeros(self.input_shape, dtype=torch.bool)
    H, W = self.input_shape
    loc = self.location
    # Very slow implementation
    for i in range(H):
      for j in range(W):
        cur = torch.tensor([i, j])
        dist = ((cur - loc)**2).sum().sqrt() 
        if dist <= self.radius:
          mask[i, j] = True
    return mask

def create_sensors(n_sensors, radius, input_shape, kind='grid', random_seed=0):
  H, W = input_shape
  assert H == W, 'Only square shapes are implemented for now'
  if kind == 'grid':
    steps = int(torch.tensor(n_sensors).sqrt().ceil())
    xx = torch.linspace(0, W, steps=steps + 1)[:steps]
    yy = torch.linspace(0, H, steps=steps + 1)[:steps]
    spatial_width = xx[1] - xx[0]
    spatial_height = yy[1] - yy[0]
    Xg, Yg = torch.meshgrid(xx + spatial_width/2, yy + spatial_height/2)
    locations = torch.stack([Xg.ravel(), Yg.ravel()]).T
    rng = torch.Generator().manual_seed(random_seed)
    idx = torch.randperm(len(locations), generator=rng)
    locations = locations[idx[:n_sensors]]
  elif kind == 'random':
    rng = torch.Generator().manual_seed(random_seed)
    locations = torch.rand(n_sensors, 2, generator=rng) * torch.tensor(input_shape)
  elif kind == 'quasi_random':
    sobol = torch.quasirandom.SobolEngine(2, scramble=True, seed=random_seed)
    locations = sobol.draw(n_sensors) * torch.tensor(input_shape)
  elif kind.startswith('barrier'):
    endpoints = {
      'barrier_v': [[0, H//2 - 1], [W - 1, H//2 - 1]], # vertical
      'barrier_h': [[W//2 - 1, 0], [W//2 - 1, H - 1]], # horizontal
      'barrier_d1': [[0, 0], [W - 1, H - 1]], # diag 1
      'barrier_d2': [[W - 1, 0], [0, H - 1]], # diag 2
    }
    # Add 1e-4 to avoid numerical instability when taking floor in case it is int - 1e-6
    x, y = (torch.tensor(endpoints[kind]) + 1e-4).chunk(2)
    u = torch.linspace(0, 1, steps=n_sensors + 1)[:n_sensors]
    u = u + (u[1] - u[0])/2 # Evenly spaced
    u = u.reshape(-1, 1)
    # Zero-indexed endpoints
    locations = u*x + (1 - u)*y
  else:
    raise ValueError(f'placement kind={kind} is not implemented')
    
  # Check and floor
  assert len(locations) == n_sensors
  locations = locations.floor() # Make at integer locations
  sensors = [
    Sensor(loc, radius, input_shape)
    for loc in locations
  ]
  return sensors


class SensorPlacementDataset(Dataset):
  def __init__(self, dataset, n_sensors, radius, kind='grid', failure_rate=0,
               return_mask=False, replace_values=(0, 0), random_seed_offset=0,
                make_cache=True, noiseless_ground_truth=False):
    self.dataset = dataset
    self.n_sensors = n_sensors
    self.radius = radius
    self.kind = kind
    self.failure_rate = failure_rate
    self.return_mask = return_mask
    self.replace_values = replace_values
    self.random_seed_offset = random_seed_offset
    self.make_cache = make_cache
    self.noiseless_ground_truth = noiseless_ground_truth
    
    if dataset[0][0].shape[-1] == 3:
        self._input_shape = dataset[0][0].shape[-3:-1]
    else:
        self._input_shape = dataset[0][0].shape[-2:]
    
    # Get unique map from replay_name to idx for rng
    self._replay_to_idx = {r: i for i, r in enumerate(self.dataset.match_metadata['static.replay_name'])}
    self._sensor_cache = dict()

  def __getitem__(self, idx):
    item = self.dataset[idx]
    assert len(item) == len(self.replace_values)
    
    # Get/create sensor placement that is fixed across replay
    md = self.dataset.metadata.iloc[idx]
    replay_name = md['static.replay_name']
    if self.make_cache and replay_name in self._sensor_cache:
      sensors, sensor_masks = self._sensor_cache[replay_name]
    else:
      match_idx = self._replay_to_idx[replay_name]
      # Fixed but unique random seed
      random_seed = match_idx + self.random_seed_offset * len(self._replay_to_idx)
      sensors = create_sensors(self.n_sensors, self.radius, self._input_shape, self.kind, random_seed=random_seed)
      # Compute masks
      sensor_masks = torch.stack([
        s.get_mask() for s in sensors
      ])
      if self.make_cache:
        # Save masks into cache
        self._sensor_cache[replay_name] = (sensors, sensor_masks)
    
    # Get mask with failures (failures are window-specific but fixed given random_seed_offset)
    # Ensure random number generator is always the same whenever this idx is called
    rng = torch.Generator().manual_seed(int(idx) + self.random_seed_offset * len(self))
    prob = (1 - self.failure_rate) * torch.ones(len(sensors))
    not_failed_mask = torch.bernoulli(prob, generator=rng).float()
    
    visible_mask = torch.conv2d(
      sensor_masks.float(), 
      not_failed_mask.view(1, not_failed_mask.shape[0], 1, 1)
    ).squeeze(0).gt(0)
    
    def replace_(x, mask, r):
      # In-place replace (not sure if the best idea, 
      #  tried to use scatter but seemed complicated)
      if x.shape[-1] == 3 or x.shape[-1] == 1:
        x[mask, ...] = r
      else:
        x[..., mask] = r
      return x
    
    if self.noiseless_ground_truth:
        # Assume the ground truth is the last location in item, apply sensor mask to everything else
        out = [
          replace_(x, ~visible_mask, r) # Replace things in the channel dimension (i.e., 0) with r
          for x, r in zip(item[:-1], self.replace_values[:-1])
        ] + [item[-1]]
    else:
        out = [
          replace_(x, ~visible_mask, r) # Replace things in the channel dimension (i.e., 0) with r
          for x, r in zip(item, self.replace_values)
        ]

    if self.return_mask:
      out += [visible_mask]
    return tuple(out)
    
  def __len__(self):
    return len(self.dataset)

# if __name__ == '__main__':
#     # Create subclass of original dataset for next window prediction
#   import sys
#   sys.path.append('..')
#   sys.path.append('../..')

#   from sc2sensor.dataset import StarCraftSensor
#   import matplotlib.pyplot as plt
#   from pathlib import Path


#   class NextWindowDataset(StarCraftSensor):

#     def __init__(self, *args, max_samples=None, **kwargs):
#         assert 'use_sparse' not in kwargs, 'Cannot set use_sparse with this dataset.'
#         assert 'compute_labels' not in kwargs, 'Cannot set use_sparse with this dataset.'
#         super().__init__(*args, use_sparse=True, compute_labels=False, **kwargs)
#         self.max_samples = max_samples
        
#         # Sort data so that next index is merely + 1
#         self.metadata = self.metadata.sort_values(['static.replay_name', 'dynamic.window_idx']).reset_index(drop=True)
#         md = self.metadata
        
#         # Get starting window indices
#         start_windows = md[(md['dynamic.num_windows'] > 1) 
#                         & (md['dynamic.window_idx'] < (md['dynamic.num_windows'] - 1))]
#         self.start_idx = start_windows.index

#     def __getitem__(self, idx):
#         # Get original indices of start and end based on input 
#         md = self.metadata
#         # Assumes sorted
#         orig_idx = self.start_idx[idx]
#         next_idx = orig_idx + 1
#         # Only sanity check first and last as this may be expensive
#         if idx == 0 or idx == len(self) - 1:
#           assert md['static.replay_name'][orig_idx] == md['static.replay_name'][next_idx], 'Replays are not the same'
#           assert md['dynamic.window_idx'][orig_idx] + 1 == md['dynamic.window_idx'][next_idx], 'Window indices are not adjacent'

#         # Get combined hyperspectral images
#         def get_hyperspectral_dense(idx):
#           # Concatenate player1 and player2 hyperspectral
#           replay_file, window_idx = self._get_replay_and_window_idx(idx)
#           with np.load(replay_file) as data:
#             player_1_hyperspectral = self._extract_hyperspectral(
#             'player_1', data, window_idx)
#             player_2_hyperspectral = self._extract_hyperspectral(
#             'player_2', data, window_idx)
#           return torch.concat([player_1_hyperspectral.to_dense(), 
#                             player_2_hyperspectral.to_dense()], 
#                             dim=-3).float()
#         windows = [get_hyperspectral_dense(idx) for idx in [orig_idx, next_idx]] 
#         x = windows[0]
#         y = windows[1] - windows[0] # Compute diff
#         return x, y

#     def __len__(self):
#         if self.max_samples is not None:
#           return min(self.max_samples, len(self.start_idx))
#         else:
#           return len(self.start_idx)


#   next_window_train = NextWindowDataset(root=str(Path('..')/'..'/'data'), subdir='starcraft-sensor-dataset', 
#                                         train=True, max_samples=1000)
#   show_idx = [0, 1, 2, 100, 200, 300]
#   nr, nc = len(show_idx), len(SUPPORTED_PLACEMENT_KINDS)
#   fig, axes_mat = plt.subplots(nr, nc, figsize=[nc*3, nr*3])
#   axes_mat = axes_mat.reshape((nr, nc)) # Ensure 2D
#   for kind, axes in zip(SUPPORTED_PLACEMENT_KINDS, axes_mat.T):
#     sp_dataset = SensorPlacementDataset(
#       next_window_train, n_sensors=10, radius=15, 
#       kind=kind, failure_rate=0, return_mask=True)
#     for idx, ax in zip(show_idx, axes):
#       ax.imshow(sp_dataset[idx][2])
#       ax.axis('off')
#       ax.set_title(f'{kind}')
#       if kind == SUPPORTED_PLACEMENT_KINDS[0]:
#         ax.set_ylabel(f'Sample {idx}')
#   fig.tight_layout()
#   plt.savefig('sensor_test.png')
#   print('Notice that first 3 are the same (i.e., in the same match) while the last 3 are different')

