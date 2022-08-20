import os
import json
import copy
import collections
import warnings

import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional
import PIL

from .modules import StarCraftToImageReducer

RACE_TO_ID = {'Terran':1, 'Zerg': 2, 'Protoss': 3}
ID_TO_RACE = {v:k for k, v in RACE_TO_ID.items()}
MAP_NAME_TO_ID = {'Acolyte LE': 0,
                  'Abyssal Reef LE': 1,
                  'Ascension to Aiur LE': 2,
                  'Mech Depot LE': 3,
                  'Odyssey LE': 4,
                  'Interloper LE': 5,
                  'Catallena LE (Void)': 6}
# grab the first 5 maps
SUBMAP_NAMES_TO_ID = {name: 2*mid if mid <= 4 else None for name, mid in MAP_NAME_TO_ID.items()}

MAP_CLASSES = (('Acolyte LE', 'Beginning'),
              ('Acolyte LE', 'End'),
              ('Abyssal Reef LE', 'Beginning'),
              ('Abyssal Reef LE', 'End'),
              ('Ascension to Aiur LE', 'Beginning'),
              ('Ascension to Aiur LE', 'End'),
              ('Mech Depot LE', 'Beginning'),
              ('Mech Depot LE', 'End'),
              ('Odyssey LE', 'Beginning'),
              ('Odyssey LE', 'End'))

_ALL_LABELS = (
    ('Terran', 'Terran', 'Win'),
    ('Terran', 'Terran', 'NotWin'),
    ('Terran', 'Zerg', 'Win'),
    ('Terran', 'Zerg', 'NotWin'),
    ('Terran', 'Protoss', 'Win'),
    ('Terran', 'Protoss', 'NotWin'),
    ('Zerg', 'Terran', 'Win'),
    ('Zerg', 'Terran', 'NotWin'),
    ('Zerg', 'Zerg', 'Win'),
    ('Zerg', 'Zerg', 'NotWin'),
    ('Zerg', 'Protoss', 'Win'),
    ('Zerg', 'Protoss', 'NotWin'),
    ('Protoss', 'Terran', 'Win'),
    ('Protoss', 'Terran', 'NotWin'),
    ('Protoss', 'Zerg', 'Win'),
    ('Protoss', 'Zerg', 'NotWin'),
    ('Protoss', 'Protoss', 'Win'),
    ('Protoss', 'Protoss', 'NotWin')
)

def _get_10_labels(race):
    return tuple([lab for lab in _ALL_LABELS if lab[0] == race or lab[1] == race])
LABELS_DICT = {k.lower() + '_10': _get_10_labels(k) for k in ['Terran', 'Zerg', 'Protoss']}
LABELS_DICT['all'] = _ALL_LABELS
DEFAULT_10 = 'zerg_10'

# Handcrafted (not used yet)
_HANDCRAFTED_TEN_LABELS = (
    # Include all that have Zerg
    #('Terran', 'Terran', 'Win'),
    #('Terran', 'Terran', 'NotWin'),
    ('Terran', 'Zerg', 'Win'),
    ('Terran', 'Zerg', 'NotWin'),
    #('Terran', 'Protoss', 'Win'),
    #('Terran', 'Protoss', 'NotWin'),
    ('Zerg', 'Terran', 'Win'),
    ('Zerg', 'Terran', 'NotWin'),
    ('Zerg', 'Zerg', 'Win'),
    ('Zerg', 'Zerg', 'NotWin'),
    ('Zerg', 'Protoss', 'Win'),
    ('Zerg', 'Protoss', 'NotWin'),
    #('Protoss', 'Terran', 'Win'),
    #('Protoss', 'Terran', 'NotWin'),
    ('Protoss', 'Zerg', 'Win'),
    ('Protoss', 'Zerg', 'NotWin'),
    #('Protoss', 'Protoss', 'Win'),
    #('Protoss', 'Protoss', 'NotWin')
)


class StarCraftSensor(torch.utils.data.Dataset):
    '''
    Given a directory `root` and `subdir` create a StarCraft dataset
    from metadata and npz files in that directory.

    Params
    ------
    `use_sparse` : If False (default), then return dense tensors
                   for unit information.  One is '{prefix}_unit_ids'
                   and the other is '{prefix}_values'. Both have
                   shape (C', W, H) where C' is the number of
                   overlapped units for one xy coordinate.
                   Importantly, this is variable for each
                   instance and thus must be padded if batched
                   together with other samples (see included 
                   dataloader for this functionality).

                   If True, then return sparse PyTorch tensors
                   with shape (C, W, H) where C is the number of
                   unit types (different for players vs neutral).

    `to_float` :   Makes all return values except '{prefix}_unit_ids'
                   float type via `float()`. 'unit_ids' will
                   remain as LongTensor so they can be used with
                   embedding layers.
    '''
    def __init__(self, root, subdir='starcraft-sensor', train=True, image_size=64, postprocess_metadata_fn=None, 
                 label_kind='all', label_func=None, use_sparse=False, to_float=True, debug=False, use_cache=True,
                 drop_na=True, compute_labels=True):
        self.data_dir = os.path.join(root, subdir)
        self.train = train
        self.use_sparse = use_sparse
        self.to_float = to_float
        self.debug = debug
        self.image_size = image_size
        self.label_kind = label_kind
        self.labels = LABELS_DICT[label_kind]
        if label_func is None:
            label_func = _default_label_func
        self.label_func = label_func

        if postprocess_metadata_fn is None:
            postprocess_metadata_fn = _postprocess_train_test_split
        
        # Load and preprocess metadata
        csv_cache_file = os.path.join(self.data_dir, 'cached-metadata.csv')
        if not os.path.exists(csv_cache_file):
            with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
                orig_metadata = json.load(f)
            md = pd.json_normalize(orig_metadata, sep='.')
            md.to_csv(csv_cache_file)
        else:
            print('Using cached CSV metadata')
            md = pd.read_csv(csv_cache_file)

        if not compute_labels:
            print('Not computing labels')
            md['computed.target_label'] = '(Unknown)'
            md['computed.target_id'] = 0
        else:
            # Add targets based on label func
            md['computed.target_label'] = md.apply(lambda row: self.label_func(row)[0], axis=1)
            md['computed.target_id'] = md.apply(lambda row: self.label_func(row)[1], axis=1)

            if drop_na:
                # dropping any entries which do not have a label
                # NOTE: any missing target ids should be set with pd.NA instead of None to avoid the computed.target_id
                # series being casted to float to account for the missing value. See below for details:
                # https://pandas.pydata.org/docs/dev/user_guide/integer_na.html#nullable-integer-data-type
                md = md.dropna(subset=['computed.target_id']).reset_index(drop=True)

        # Filter metadata to get different windows
        print('Post-processing metadata')
        temp_match_md = md.drop_duplicates(subset=['static.replay_name'])
        md = postprocess_metadata_fn(md, temp_match_md, train=self.train, labels=self.labels)
        md = md.reset_index(drop=True)  # Renumber rows

        # Save metadata and match_metadata after post processing
        self.metadata = md
        self.match_metadata = md.drop_duplicates(subset=['static.replay_name']).reset_index(drop=True)

	# Load cache
        train_test = ('train' if self.train else 'test')
        dataset_name = f'{self.__class__.__name__}_{train_test}' 
        cache_filename = os.path.join(self.data_dir, dataset_name + '.npz')
        if use_cache and os.path.exists(cache_filename):
            print(f'Using cached data at {cache_filename}')
            with np.load(cache_filename) as cache:
                # Extract from compressed archive
                self._cache = dict(x=cache['x'], y=cache['y'])
        else:
            self._cache = None
        print('Finished dataset init')

    def __str__(self):
        item = self[0]
        
        out = '-----------------\n'
        out += f'{self.__class__.__name__}\n'
        out += f'  data_dir = {self.data_dir}\n'
        out += f'  train = {self.train}\n'
        out += f'  num_windows = {len(self)}\n'
        out += f'  num_matches = {self.num_matches()}\n'
        out += f'  image_size = ({self.image_size}, {self.image_size})\n'
        out += f'  label_kind = {self.label_kind}\n'
        out += f'  num_labels = {len(self.labels)}\n'
        #out += f'  labels = {self.labels}\n'
        if type(item) is tuple:
            out += f'  getitem = {self[0]}\n'
        elif type(item) is dict:
            out += f'  getitem_keys = {self[0].keys()}\n'
        else:
            out += f'  getitem_type = {type(self[0])}\n'
        out += '-----------------'
        return out
        
    def __len__(self):
        return len(self.metadata)

    def num_matches(self):
        return len(self.match_metadata)
    
    def __getitem__(self, idx):
        if self.use_sparse:
            return self._sparse_getitem(idx)
        else:
            return self._dense_getitem(idx)

    def _sparse_getitem(self, idx):
        replay_file, window_idx = self._get_replay_and_window_idx(idx)
        with np.load(replay_file) as data:
            data_dict = dict(
                # Extract hyperspectral
                player_1_hyperspectral = self._extract_hyperspectral(
                    'player_1', data, window_idx),
                player_2_hyperspectral = self._extract_hyperspectral(
                    'player_2', data, window_idx),
                neutral_hyperspectral = self._extract_hyperspectral(
                    'neutral', data, window_idx),
                **self._get_non_unit(data, idx, window_idx),
            )
        return self._check_float(data_dict)

    def _get_non_unit(self, data, idx, window_idx):
        # Tuples of strings can be used in dictionaries
        smd = self.metadata.iloc[idx]  # Single metadata entry
        return dict(
            # Extract tabular
            player_1_tabular = self._extract_other(
                'player_1', 'tabular', data, window_idx),
            player_2_tabular = self._extract_other(
                'player_2', 'tabular', data, window_idx),
            # Extract map_state
            player_1_map_state = self._extract_other(
                'player_1', 'map_state', data, window_idx),
            player_2_map_state = self._extract_other(
                'player_2', 'map_state', data, window_idx),
            pathing_grid = self._extract_dense_mat('pathing_grid', data),
            terrain_height = self._extract_dense_mat('terrain_height', data),
            placement_grid = self._extract_dense_mat('placement_grid', data),
            # Extract target information
            is_player_1_winner = self._get_is_player_1_winner(idx),
            target_label = smd['computed.target_label'],
            target_id = smd['computed.target_id'],
        )

    def _dense_getitem(self, idx):
        # Get necessary metadata
        replay_file, window_idx = self._get_replay_and_window_idx(idx)
        with np.load(replay_file) as data:
            data_dict = dict(
                # Extract units
                **self._extract_dense_unit_repr('player_1', data, window_idx),
                **self._extract_dense_unit_repr('player_2', data, window_idx),
                **self._extract_dense_unit_repr('neutral', data, window_idx),
                **self._get_non_unit(data, idx, window_idx),
            )
        return self._check_float(data_dict)

    def _check_float(self, data_dict):
        def _try_float(k, v):
            if k.endswith('unit_ids'):
                return v
            try:
                v = v.float()
            except AttributeError:
                pass  # Ignore if not tensor
            return v
        if self.to_float:
            # Preserve LongTensor of unit_ids tensors otherwise convert
            return {k:_try_float(k, v) 
                    for k, v in data_dict.items()}
        else:
            return data_dict

    def _get_replay_and_window_idx(self, idx):
        item = self.metadata.iloc[idx]
        base = os.path.splitext(item['static.replay_name'])[0]
        replay_file = os.path.join(self.data_dir, 'replay_files' , f'{base}.npz')
        window_idx = item['dynamic.window_idx']
        return replay_file, window_idx

    def _get_is_player_1_winner(self, idx):
        item = self.metadata.iloc[idx]
        return torch.tensor(item['static.replay_info.player_stats.player_1.result'] == 'Win')

    def _extract_hyperspectral(self, prefix, data, window_idx):
        shape = data[f'{prefix}_hyperspectral_shape']
        if len(shape) == 4:
            shape = shape[1:] # Extract only last 3 dimensions as there is one batch dimension
        indices = data[f'{prefix}_hyperspectral_window_{window_idx}_indices']
        values = data[f'{prefix}_hyperspectral_window_{window_idx}_values']
        indices, values, shape = self._resize_hyper(indices, values, shape)
        return torch.sparse_coo_tensor(indices, values, tuple(shape)).coalesce()

    def _resize_hyper(self, indices, values, shape):
        if self.image_size == 64:
            return indices, values, shape
        assert self.image_size <= 64, 'image_size must be less than or equal to 64'
        scale = float(self.image_size) / 64.0

        # Convert from index to location, then scale, then floor to get new index
        orig_indices = indices # For debugging
        indices = indices.T  # Change shape to (n_nonzero, ndim) for easier inspection
        indices[:, 1:] = np.floor(scale * (indices[:, 1:] + 0.5)).astype(np.uint8)
    
        # How to handle overlaps (max coalesce - take max value for overlaps---maybe just for loop it over dictionary...and cache)
        # First sort (descending) by indices and then values
        #sort_idx = np.flip(np.lexsort((values, indices[:, 2], indices[:, 1], indices[:, 0]))) # Sort descending
        sort_idx = np.flip(np.argsort(values))  # Sort descending
        indices = indices[sort_idx, :]
        values = values[sort_idx]

        # Do unique over indices 
        unique_indices, unique_first_idx = np.unique(indices, axis=0, return_index=True)
        # Get corresponding values (which are the max because of sorting)
        unique_values = values[unique_first_idx]

        # Replace image_size in shape
        shape = (shape[0], self.image_size, self.image_size)

        # Convert to sparse tensor and coalesce, then extract new indices and values
        temp_sparse = torch.sparse_coo_tensor(unique_indices.T, unique_values, shape).coalesce()
        return temp_sparse.indices(), temp_sparse.values(), shape

    def _extract_other(self, prefix, name, data, window_idx):
        return self._extract_dense_mat(f'{prefix}_{name}_window_{window_idx}', data)

    def _extract_dense_mat(self, name, data):
        x = torch.from_numpy(data[name])
        if x.ndim == 1:
            return x

        if self.image_size == 64:
            return x
        # Using antialias because values are continuous rather than discrete here
        #  because we already aggregated over time
        x = torchvision.transforms.functional.resize(
            x.unsqueeze(0), (self.image_size, self.image_size), antialias=True).type(torch.uint8).squeeze(0)
        return x

    def _extract_dense_unit_repr(self, prefix, data, window_idx):
        shape = data[f'{prefix}_hyperspectral_shape']
        if len(shape) == 4:
            shape = shape[1:] # Extract only last 3 dimensions as there is one batch dimension
        indices = data[f'{prefix}_hyperspectral_window_{window_idx}_indices']
        values = data[f'{prefix}_hyperspectral_window_{window_idx}_values']
        indices, values, shape = self._resize_hyper(indices, values, shape)
        assert indices.shape[0] == 3, 'Should be 3 dimensional (C, W, H)'
        
        if values.shape[0] == 0:
            # if the given hyperspectral image has no nonzero elements (i.e. no values)
            base_shape = (1, *shape[-2:]) # 1 channel with of image size
            unit_ids = -torch.ones(base_shape, dtype=torch.from_numpy(indices).dtype)
            unit_values = torch.zeros(base_shape, dtype=torch.from_numpy(values).dtype)
        else:
            if self.debug:
                orig_data = np.hstack([indices.T, values.reshape((-1, 1))]).copy()
            # Group by operation from https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
            # a = a[a[:, 0].argsort()]  # Sort by first column so split indices are known
            # np.split(a[:,1], np.unique(a[:, 0], return_index=True)[1][1:]) # Use first indexes to split

            # Sort by xy coordinates so that np.split can be used later
            indices = indices.T  # Change shape to (n_nonzero, ndim) for easier inspection
            sort_idx = np.lexsort((indices[:, 2], indices[:, 1]))
            indices = indices[sort_idx, :]
            values = values[sort_idx]

            # Now can group by via splitting on first indices as returned by
            #  the "index" output of np.unique
            sorted_xy = indices[:, 1:]
            sorted_id = indices[:, 0]
            unique_xy, unique_first_idx = np.unique(sorted_xy, axis=0, return_index=True)
            ids_group_by_xy = np.split(sorted_id, unique_first_idx[1:])
            values_group_by_xy = np.split(values, unique_first_idx[1:])

            # Create new indices + values matrix with values of
            #  c', w, h, t, v  where c' are new channels (<= num_overlap), 
            #  w is x coord, h is y coord, t is unit type id, v is timestamp value
            def create_data(xy, ids, vals):
                channel_idx = np.arange(len(ids)).reshape((-1, 1))  # New channel dimension
                xy_idx = np.broadcast_to(xy, (len(ids), 2))  # xy dimensions replicated
                ids = ids.reshape((-1, 1)) # Types (which will serve as values in the dense type tensor)
                vals = vals.reshape((-1, 1))
                return np.hstack([channel_idx, xy_idx, ids, vals])
            
            new_data = np.vstack([
                create_data(xy, ids, vals)
                for xy, ids, vals in zip(unique_xy, ids_group_by_xy, values_group_by_xy)
            ])
            
            dense_shape = (new_data[:, 0].max() + 1, *shape[-2:])  # C' x W x H
            
            unit_ids = torch.sparse_coo_tensor(new_data[:,:3].T, new_data[:,3], size=dense_shape)
            unit_ids = unit_ids.to_dense()
            unit_values = torch.sparse_coo_tensor(new_data[:,:3].T, new_data[:,4], size=dense_shape)
            unit_values = unit_values.to_dense()
            
            # Debug
            if False:
                assert orig_data.shape[0] == new_data.shape[0], 'Should have same number of nonzeros'
                def rowsort(A):
                    return A[np.lexsort(np.fliplr(A).T)]
                orig_data = rowsort(orig_data)
                rec_orig_data = new_data[:, [3, 1, 2, 4]]
                rec_orig_data = rowsort(rec_orig_data)
                print(orig_data)
                print(rec_orig_data)
                assert np.all(orig_data == rec_orig_data), 'Reconstruction is not correct'
                print(new_data)
                
        # Return ids and values
        return {f'{prefix}_unit_ids': unit_ids.long(), 
                f'{prefix}_unit_values': unit_values}


class SensorHyper(StarCraftSensor):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)


class _SensorSimpleBase(StarCraftSensor):
    def __init__(self, root, transform=None, target_transform=None, **kwargs):
        assert 'use_sparse' not in kwargs, 'use_sparse cannot be changed for SensorCIFAR10'
        assert 'to_float' not in kwargs, 'for simple datasets, use transform = torchvision.transforms.ToTensor()'
        super().__init__(root, use_sparse=False, **kwargs)
        self._reduce_to_image = StarCraftToImageReducer()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        if self._cache is not None:
            x = torch.from_numpy(self._cache['x'][idx])  # Should load np.uint8 and cast to tensor
            target = self._cache['y'][idx]
        else:
            x, target = self._get_x_and_target(idx)

        # Return as PIL image
        if x.shape[0] == 1:  # Grayscale
            img = PIL.Image.fromarray(x.squeeze(0).type(torch.uint8).numpy(), mode='L')
        elif x.shape[0] == 3:  # RGB
            img = PIL.Image.fromarray(x.permute(1,2,0).type(torch.uint8).numpy(), mode='RGB')
        else:
            raise RuntimeError('x should have 1 or 3 channels')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _get_x_and_target(self, idx):
        # Get hyperspectral dictionary
        d = super().__getitem__(idx)
        # Create batch of size 1 and then extract output
        x = self._reduce_to_image(sc_collate([d]))[0]
        # Get target id directly from already extracted dictionary
        target = d['target_id']
        return x, target


class SensorCIFAR10(_SensorSimpleBase):
    def __init__(self, root, label_kind=DEFAULT_10, **kwargs):
        assert 'image_size' not in kwargs, 'Image size is fixed to 32 for SensorCIFAR10'
        assert 'postprocess_metadata_fn' not in kwargs, 'Postprocess function cannot be changed for SensorCIFAR10'
        super().__init__(root, label_kind=label_kind, image_size=32, postprocess_metadata_fn=_postprocess_cifar10, **kwargs)


class SensorMNIST(_SensorSimpleBase):
    def __init__(self, root, label_kind=DEFAULT_10, **kwargs):
        assert 'image_size' not in kwargs, 'Image size is fixed to 28 for SensorMNIST'
        assert 'postprocess_metadata_fn' not in kwargs, 'Postprocess function cannot be changed for SensorMNIST'
        super().__init__(root, label_kind=label_kind, image_size=28, postprocess_metadata_fn=_postprocess_mnist, **kwargs)

    def _get_x_and_target(self, idx):
        x, target = super()._get_x_and_target(idx)

        # Reorder RGB to get in terms of player 1, player 2 and neutral
        #  and get corresponding masks
        x_ordered = x[[2,0,1], :, :] / 255.0
        m1, m2, mn = (x_ordered > 0)

        # Normalize neutral
        n = x_ordered[2, :, :]
        x_ordered[2, :, :] = (n - n.min()) / (n.max() - n.min())

        # Rescale values
        # NOTE: for player 2 it is 0.4 to 0 (so it becomes a negative multiplier)
        scales = np.array([[0.55, 1], [0.45, 0], [0.48, 0.52]])
        v1, v2, vn = [(sc[0] + (sc[1] - sc[0]) * v) for v, sc in zip(x_ordered, scales)]

        new_x = (vn * (~m2) + v2 * m2) * (~m1) + v1 * m1
        return 255.0 * new_x.unsqueeze(0), target


# Label func
def make_map_plus_begin_end_game_label(smd):
    # Input is single metadata row as pandas row
    # Output is target_label (e.g., string) and target_id (e.g., 0-9)
    map_id = SUBMAP_NAMES_TO_ID[smd['static.game_info.map_name']]
    is_end = (smd['dynamic.window_idx'] / smd['dynamic.num_windows']) > 0.5
    if map_id is not None:
        target_id = map_id + is_end  # int + True == int + 1 and int + False == int
        target_label = (smd['static.game_info.map_name'], 'End' if is_end else 'Beginning')
    else:
        target_id = pd.NA
        target_label = pd.NA
    return target_label, target_id

def _default_label_func(smd):
    return make_map_plus_begin_end_game_label(smd)
    # label_to_id = {label:i for i, label in enumerate(obj.labels)}
    # target_label = (
    #     ID_TO_RACE[smd['static.game_info.player_info.player_1.race_actual']],
    #     ID_TO_RACE[smd['static.game_info.player_info.player_2.race_actual']],
    #     'Win' if smd['static.replay_info.player_stats.player_1.result'] == 'Win' else 'NotWin',
    # )
    # target_id = label_to_id[target_label]
    # return target_label, target_id


def _postprocess_train_test_split(metadata, match_metadata, train, labels, perc_train=0.9, random_state=1):
    # First filter to only matches that are in the labels
    metadata = pd.concat([filt_md for target_id, filt_md, filt_mmd in _stratify_by_label(metadata)])
    match_metadata = metadata.drop_duplicates(subset=['static.replay_name']).reset_index(drop=True)

    # Split into train and test along unique matches
    n_match_train = int(np.round(perc_train * len(match_metadata)))
    perm = np.random.RandomState(random_state).permutation(len(match_metadata))
    #print('First 5 of train_test permutation', perm[:5])
    if train == 'all':
        matches = match_metadata
    elif train == True:
        matches = match_metadata.iloc[perm[:n_match_train], :]
    elif train == False:
        matches = match_metadata.iloc[perm[n_match_train:], :]
    else:
        raise ValueError('`train` must be True, False or \'all\'')
    # Filter based on matches
    return _filter_by_matches(metadata, matches).sample(frac=1, random_state=0).reset_index(drop=True)  # Shuffle


def _filter_by_matches(metadata, match_metadata):
    metadata = metadata[metadata['static.replay_name'].isin(match_metadata['static.replay_name'])]
    return metadata.reset_index(drop=True)


def _postprocess_cifar10(*args, **kwargs):
    N_TRAIN_CIFAR10 = 5000
    N_TEST_CIFAR10 = 1000
    return _postprocess_simplified(
        *args, **kwargs,
        n_train=N_TRAIN_CIFAR10, n_test=N_TEST_CIFAR10)


def _postprocess_mnist(*args, **kwargs):
    N_TRAIN_MNIST = 6000
    N_TEST_MNIST = 1000
    return _postprocess_simplified(
        *args, **kwargs,
        n_train=N_TRAIN_MNIST, n_test=N_TEST_MNIST)


def _postprocess_simplified(metadata, match_metadata, train, labels, n_train, n_test):
    '''Filter metadata via stratified sampling. 
    First stratify based on class. 
    Then split based on matches. 
    Finally, sample without replacement to get exact numbers.'''
    return pd.concat([
        _train_test_split_and_sample(
            filt_md, filt_mmd, train, None, # labels is not used...
            n_train=n_train, n_test=n_test, random_state=int(target_id))
        for target_id, filt_md, filt_mmd in _stratify_by_label(metadata)
    ]).sample(frac=1, random_state=0).reset_index(drop=True)  # Shuffle rows


def _stratify_by_label(md):
    # Get unique target ids
    unique_ids = md['computed.target_id'].unique()
    # Get metadata filtered by target id
    for target_id in unique_ids:
        # Filter by target_id
        filt_md = md[md['computed.target_id'] == target_id].reset_index(drop=True)
        # Get match metadata (i.e., first unique occurences)
        filt_mmd = filt_md.drop_duplicates(subset=['static.replay_name']).reset_index(drop=True)
        # Yield this group
        yield target_id, filt_md, filt_mmd


def _train_test_split_and_sample(md, filt_mmd, train, labels, n_train, n_test, random_state=0):
    '''Split into train and test based on match data. Then sample to exact n_train or n_test.'''
    # Split roughly into train and test by matches
    perc_train = n_train / (n_train + n_test)
    # NOTE: This returns train or test metadata already
    md = _postprocess_train_test_split(md, filt_mmd, train, labels, perc_train=perc_train)

    # Randomly sample exact number of windows
    perm = np.random.RandomState(random_state).permutation(len(md))
    if train == 'all':
        raise ValueError('For sensorMNIST and SensorCIFAR10 train=\'all\' is not an option')
    elif train == True:
        md = md.iloc[perm[:n_train], :]
    elif train == False:
        md = md.iloc[perm[:n_test], :]
    else:
        raise ValueError('`train` must be True, False or \'all\'')
    # print(len(filt_mmd), len(md))
    return md.reset_index(drop=True)


def starcraft_dense_ragged_collate(batch):
    '''
    Function to be passed as `collate_fn` to torch.utils.data.DataLoader
    when using use_sparse=False (default) for StarCraftSensor.
    This handles padding the dense tensors so they have the same shape
    in each batch.

    `sc_collate` is an alias for this function as well.

    Example:
    >>> scdata = StarCraftSensor(root, use_sparse=False)
    >>> torch.utils.data.DataLoader(scdata, collate_fn=sc_collate, batch_size=32, shuffle=True)
    '''
    elem = batch[0]
    elem_type = type(elem)
    assert isinstance(elem, collections.abc.Mapping), 'Only works for dictionary-like objects'
    
    def pad_as_needed(A, n_target_channels):
        channel_pad = n_target_channels - A.shape[0]
        if channel_pad > 0:
            #A = np.pad(A, ((0, channel_pad), (0, 0), (0, 0)), mode='minimum')
            A = torch.nn.functional.pad(A, ((0, 0, 0, 0, 0, channel_pad)), value=A.min())
        return A
    
    def collate_pad(batch_list):
        # Pad each to have the same number of first dimension (i.e. channels for hyperspectral)
        try:
            ndim = batch_list[0].ndim
        except AttributeError:
            ndim = 0 # For non-tensors
        if ndim > 0:
            unique_channels = np.unique([d.shape[0] for d in batch_list], axis=0)
            if len(unique_channels) > 1:
                n_target_channels = unique_channels.max()
                batch_list = [pad_as_needed(d, n_target_channels) for d in batch_list]
        return torch.utils.data.dataloader.default_collate(batch_list)
    
    return elem_type({key: collate_pad([d[key] for d in batch]) for key in elem}) 


# Shorter alias for starcraft_dense_ragged_collate
sc_collate = starcraft_dense_ragged_collate
