import sys
from pathlib import Path
from warnings import warn
import time
import json

import numpy as np
import sparse as np_sparse
import torch
from pysc2.lib import features
from pysc2.lib import stopwatch
from pysc2.lib.point import Point

from skimage.transform import resize

sw = stopwatch.sw

from .utils.unit_type_data import NEUTRAL_IDS as neutral_ids, NONNEUTRAL_IDS as nonneutral_ids, UNKNOWN_CHANNEL

PLAYER_COMMON_KEYS = ['player_id', 'minerals', 'vespene', 
                      'food_used', 'food_cap', 'food_army', 'food_workers', 
                      'idle_worker_count', 'army_count', 'warp_gate_count', 
                      'larva_count']

class DataExtractor():
    """A class for extracting a dataset from a **single** replay file."""
    def __init__(self, game_info, sc2_flags, replay_info):
        self.game_info = game_info
        self.sc2_flags = sc2_flags
        self.replay_info = replay_info
        self.path_to_replay = Path(sc2_flags.replay)

        self.obs_counter = 0  # a counter to keep track of the number of observations we see
        self.player_obs_counters = dict()  # used for confirming the player_X extraction happens properly
        self.summarize_step = sc2_flags.summarize_mul  # the number of frames to summarize over
        self.num_frames = replay_info.game_duration_loops
        # check if replay is long enough, and if not, raise error which can be handled in extraction script
        assert  self.num_frames >= 2*self.summarize_step, \
                    f'The given replay is too short with {self.num_frames} frames and a step of {self.summarize_step}.'

        # self._feature_data = []
        self._metadata = []
        self._raw_data_dict = {'hyperspectral': []}
        self.unit_count = []

        # setting up attributes for raw_data extraction
        self.raw_image_size = sc2_flags.raw_image_size
        # setting raw_map_size to only be within the playable area
        self.playable_area_0 = game_info.start_raw.playable_area.p0
        self.playable_area_1 = game_info.start_raw.playable_area.p1
        self.raw_map_size = Point(self.playable_area_1.x - self.playable_area_0.x,
                                  self.playable_area_1.y - self.playable_area_0.y)

        self.raw_map_max_dim = max(self.raw_map_size.x, self.raw_map_size.y)
        # conversions from raw_unit coordinates
        self.raw_width_to_hyper_multiplier = self.raw_image_size[0] / self.raw_map_max_dim
        self.raw_height_to_hyper_multiplier = self.raw_image_size[1] / self.raw_map_max_dim

        # defining a mapping from unit types to a channel idx for hyperspectral images
        # we will have a mapping for neutral and for nonneutral utils
        self.nonneutral_ids_to_channel = {unit_type: channel_idx for channel_idx,
                                                                     unit_type in enumerate(nonneutral_ids)}

        self.neutral_id_to_channel = {unit_type: channel_idx for channel_idx, unit_type in enumerate(neutral_ids)}
        # each player gets their own set of channels for each unit type
        self.player_type_to_channel_offsets = {1:    0,  # 1 := self player
                                               2:    len(nonneutral_ids),  # 2  := enemy players
                                               16: 2*len(nonneutral_ids),  # 16 := neutral players (i.e. materials)
                                        }    # used to find the right channel index based off player_type and unit_type
        n_hyperspectral_channels = 2 * len(nonneutral_ids) + len(neutral_ids)
        self.hyperspectral_image_size = (n_hyperspectral_channels, *self.raw_image_size)

        self.static_metadata = self.extract_static_metadata()

    @sw.decorate
    def add_unit_count(self, num):
        self.unit_count.append(num)

    
    @sw.decorate
    def extract_and_append_data(self, obs, step):
        self.obs_counter += 1
        self.extract_raw_features(obs)  # returns None since it appends to self._raw_data_dict

        metadata_dict = self.extract_metadata(obs, step)
        self._metadata.append(metadata_dict)
        # self._feature_data.append(self.extract_features(obs))
        return None

    @sw.decorate
    def extract_and_append_just_player_data(self, obs, step, player_id):
        if self.player_obs_counters.get(player_id) is None:
            # initialize player_id data in the raw_dict
            self.player_obs_counters[player_id] = -1  # initializes to -1 since it will be incremented later
            self._raw_data_dict[f'player_{player_id}_map_state'] = []
            self._raw_data_dict[f'player_{player_id}_tabular'] = []

        self.player_obs_counters[player_id] += 1
        self._extract_just_player_raw_features(obs, player_id)
        # checking we are still in step with the main extraction
        main_step = self._metadata[self.player_obs_counters[player_id]]['dynamic']['frame_idx']
        assert main_step == step, f'Extraction mismatch for player_{player_id} and main'
        return None
        
    @sw.decorate
    def process_extraction(self):
        """Post processing of the data extracted from the single replay extraction (e.g., summarizing data)"""
        if len(self.player_obs_counters) != 0:
            for player_id, obs_count in self.player_obs_counters.items():
                # note: player_obs_counter starts at -1, so need to increment by +1 before checking
                assert (obs_count+1) == self.obs_counter, f'Extraction mismatch between main and player_{player_id}'
        # The replay has ended, summarize the data
        self._post_process_raw()  # merging and summarizing raw data
        self._summarize_metadata()
        # merged_dict = self._merge()  # merging the feature data
        # merged_dict = self._post_process(merged_dict)
        # merged_dict['screen_unit_count'] = self.unit_count  # Add unit count info
        # self._feature_data = merged_dict  # overwriting the feature data dict with the merged dict
        return None

    @sw.decorate
    def save(self, save_directory, checkpoint_metadata=True):
        save_dir = Path(save_directory)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # saving raw data
        self._save_raw(save_dir)  # splitting into player+window specific dicts and saving
        if checkpoint_metadata:
            # saving metadata to a temporary directory
            self._checkpoint_metadata(save_dir)

        # print(f'Arrays to be saved: {list(self._feature_data.keys()) + list(self._raw_data_dict.keys())}')
        # print(f'Saving extracted data to: {str(save_dir)}')
        # np.savez_compressed(save_dir/'feature_data.npz', **self._feature_data)
        # saving a config file in case we want to look back at the data extraction process 
        # self._save_config(save_dir)
        
        print(f'Successfully saved extracted data')
        return None



    @sw.decorate
    def _summarize_metadata(self):
        n_per_window = self.summarize_step
        # Note: _metadata has already been clipped in _merge_and_clip_raw
        assert len(self._metadata) % n_per_window == 0, 'Should be equally divisible'
        n_windows = len(self._metadata) // n_per_window
        # Extract the metadata for last frame
        self._metadata = self._metadata[(n_per_window-1)::n_per_window]
        # Add dynamic metadata for windows
        for ei, m in enumerate(self._metadata):
            m['dynamic']['window_idx'] = ei
            m['dynamic']['num_windows'] = n_windows
            m['dynamic']['window_percent'] = ei / n_windows
        assert len(self._metadata) == n_windows, 'Bug in slicing of metadata'
        return None

    @sw.decorate
    def _save_raw(self, save_directory):
        save_dir = Path(save_directory)
        n_windows = len(self._raw_data_dict['hyperspectral'])
        assert n_windows == (self.obs_counter // self.summarize_step), \
                                f'Bug in summarization. {n_windows, self.obs_counter, self.summarize_step}'
        # extracting and saving player specific raw data
        replay_dict = dict()
        # recording replay specific, but not window specific info
        # Invert pathing grid so 1 corresponds to pathable (not sure why it is this way
        replay_dict['pathing_grid'] = ~self.extract_crop_and_resize_map_data(self.game_info.start_raw.pathing_grid,
                                                                             pad_value=1).astype(bool)
        replay_dict['placement_grid'] = self.extract_crop_and_resize_map_data(self.game_info.start_raw.placement_grid).astype(bool)

        # Normalize terrain height to be between 0 and 255 (np.uint8)
        replay_dict['terrain_height'] = self.extract_crop_and_resize_map_data(self.game_info.start_raw.terrain_height)

        player_ids = list(self.player_type_to_channel_offsets)
        # looping over players to save player specific information
        for player_id, next_player_id in zip(player_ids, (player_ids+[None])[1:]):
            # slicing to just the channels that correspond to the current player
            just_player_channel_slice = slice(self.player_type_to_channel_offsets[player_id], 
                                  self.player_type_to_channel_offsets.get(next_player_id)) 
            player_hyperspectral = self._raw_data_dict['hyperspectral'][:, just_player_channel_slice, ...]

            if player_id != 16:  
                # not a neutral player, so grab the creep and visibility
                player_name = f'player_{player_id}'  # used for saving later
                assert player_hyperspectral.shape[1] == len(nonneutral_ids), 'Incorrect channel shape'
            else:
                # a neutral player
                player_name = 'neutral'  # used for saving later
                assert player_hyperspectral.shape[1] == len(neutral_ids), 'Incorrect channel shape'

            # recording player specific, but not window specific info
            replay_dict[f'{player_name}_hyperspectral_shape'] = player_hyperspectral.shape

            # looping over windows and saving each player+window specific information
            for window_idx in range(n_windows):
                window_hyperspectral = player_hyperspectral[window_idx]
                replay_dict[f'{player_name}_hyperspectral_window_{window_idx}_values'] = window_hyperspectral.data
                replay_dict[f'{player_name}_hyperspectral_window_{window_idx}_indices'] = window_hyperspectral.coords

                if player_id != 16:
                    # if not neutral, record map state for that player and tabular info for that player
                    map_state = self._raw_data_dict.get(f'{player_name}_map_state')
                    if map_state is not None:
                        replay_dict[f'{player_name}_map_state_window_{window_idx}'] = map_state[window_idx]
                    tabular = self._raw_data_dict.get(f'{player_name}_tabular')
                    if tabular is not None:
                        replay_dict[f'{player_name}_tabular_window_{window_idx}'] = tabular[window_idx]
            
            # saving
            replay_save_dir = save_dir/'replay_files'
            if not replay_save_dir.exists():
                replay_save_dir.mkdir()
            np.savez_compressed(replay_save_dir/(self.path_to_replay.with_suffix('.npz').name), **replay_dict)

        return None

    @sw.decorate
    def _checkpoint_metadata(self, save_dir):
        """Saves the metadata to a temporary directory"""
        checkpoint_dir = Path(save_dir) / 'temp_metadata_dir'
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir()
        with open(checkpoint_dir/f'{self.path_to_replay.with_suffix("").name}_metadata.json', 'w') as fh:
            json.dump(self._metadata, fh, sort_keys=True)

    # @sw.decorate
    # def _save_config(self, save_dir):
    #     """Saving a config file recording the setup for the data extraction process.
    #     Will save: `game_info` and the *key* `FLAGS` specified when extracting"""
    #     save_dir = Path(save_dir)
    #     self.game_info.start_raw.pathing_grid.data = b''    # removing useless byte data
    #     self.game_info.start_raw.terrain_height.data = b''  # removing useless byte data
    #     self.game_info.start_raw.placement_grid.data = b''  # removing useless byte data
    #     game_info_str = 'v Game info v'.center(60, '-') + '\n' + str(self.game_info) + \
    #                     '\n' +'^ Game info ^'.center(60, '-')
        
    #     # extracts the FLAGS which were set during the extraction script
    #     # edited from: https://github.com/abseil/abseil-py/issues/92#issuecomment-467608419
    #     key_flags = self.sc2_flags.get_key_flags_for_module(sys.argv[0])
    #     flag_info_str = 'v Extraction Flag info v'.center(60, '-') + '\n' + \
    #                     '\n'.join(f.serialize() for f in key_flags) + '\n' + '^ Extraction Flag info ^'.center(60, '-')

    #     replay_info_str = 'v Replay info v'.center(60, '-') + '\n' + str(self.replay_info) + \
    #                       '\n' +'^ Replay info ^'.center(60, '-')

    #     with open(save_dir/'config.txt', 'w') as config:
    #         config.writelines(f'Data extracted on: {time.strftime("%m %d %Y   %H %M %S ")}')
    #         config.writelines('\n'*3)
    #         config.writelines(flag_info_str)
    #         config.writelines('\n'*3)
    #         config.writelines(game_info_str)
    #         config.writelines('\n'*3)
    #         config.writelines(replay_info_str)
    #     return None

    @sw.decorate
    def extract_static_metadata(self):
        def leaf(obj, attr):
            return {k: getattr(obj, k) for k in attr}
        def player_info(obj):
            return leaf(obj, ['player_id','race_requested','race_actual'])
        def xy(obj):
            return leaf(obj, ['x','y'])
        def map_item(obj):
            map_item = leaf(obj, ['bits_per_pixel'])
            map_item['size'] = xy(obj.size)
            return map_item
        def playable_area(obj):
            return dict(p0=xy(obj.p0), p1=xy(obj.p1))
        def start_locations(obj):
            return xy(obj)
        def options(obj):
            return leaf(obj, ['raw','score'])
        def mod_names(obj):
            return list(obj)
        def nest_player_info(info):
            return {
                ('player_' + str(player['player_id'])): {
                    k.replace('player_', ''):v
                    for k, v in player.items() if k != 'player_id'
                }
                for player in info
            }
        def start_raw(obj):
            return dict(
                map_size = xy(obj.map_size),
                pathing_grid = map_item(obj.pathing_grid),
                terrain_height = map_item(obj.terrain_height),
                placement_grid = map_item(obj.placement_grid),
                playable_area = playable_area(obj.playable_area),
                start_locations = xy(obj.start_locations[0]),
            )
        def serialize_game_info(obj):
            return dict(
                map_name = obj.map_name,
                player_info = nest_player_info([player_info(subobj) for subobj in obj.player_info]),
                start_raw = start_raw(obj.start_raw),
                options = options(obj.options),
                mod_names = mod_names(obj.mod_names),
            )

        # Replay-specific
        # From https://blizzard.github.io/s2client-api/sc2__gametypes_8h_source.html
        result_int_to_result = {1:'Win', 2:'Loss', 3:'Tie', 4:'Undecided'}
        def player_stats(obj):
            def stats(obj):
                assert obj.player_info.player_id == obj.player_result.player_id
                return dict(
                    player_id = obj.player_result.player_id,
                    player_result_int = obj.player_result.result,
                    player_result = result_int_to_result[obj.player_result.result],
                    player_mmr = obj.player_mmr,
                    player_apm = obj.player_apm,
                )
            return nest_player_info([stats(player) for player in obj])
        def serialize_replay_info(obj):
            # Skipping things that are in game_info such as map_name
            info = leaf(obj, ['game_duration_loops', 'game_duration_seconds', 
                              'game_version', 'data_build', 'base_build', 'data_version'])
            info['game_fps_calculated'] = info['game_duration_loops']/info['game_duration_seconds']
            info['player_stats'] = player_stats(obj.player_info)
            return info

        static_metadata = dict(
            game_info = serialize_game_info(self.game_info),
            replay_info = serialize_replay_info(self.replay_info),
            replay_name = self.path_to_replay.name,
            extracted_image_size = self.raw_image_size,
            num_frames_per_window = self.summarize_step,
        )
        return static_metadata

    @sw.decorate
    def extract_metadata(self, obs, step):
        date_time_format = '%Y-%m-%d_%H-%M-%S'
        dynamic = dict(
            # Per frame metadata
            frame_idx = step, # step
            # frame_percent = (step + 1) / self.num_frames, # NOTE: num_frames may be different than n_observations
            date_time_str = time.strftime(date_time_format),
            date_time_format = date_time_format,
            timestamp = time.time(),
        )

        metadata = dict(static=self.static_metadata, dynamic=dynamic)
        return metadata

    @sw.decorate
    def get_metadata(self):
        return self._metadata

    @sw.decorate
    def extract_features(self, obs):
        # NOTE Looking at transform_obs() func especially lines 1151 and following from
        # https://github.com/deepmind/pysc2/blob/master/pysc2/lib/features.py
        # can be very useful for determining which features are available and how to extract them

        # Extract all minimap features
        minimap_features = {
            'minimap_' + f.name: f.unpack(obs.observation)
            for f in features.MINIMAP_FEATURES
        }
        screen_features = {
            'screen_' + f.name: f.unpack(obs.observation)
            for f in features.SCREEN_FEATURES
        }
        other_features = dict(
            dead_units=len(obs.observation.raw_data.event.dead_units)
        )
        # Combine all feature dictionaries (in Python 3.5 or greater)
        all_features = {**minimap_features, **screen_features, **other_features}
        # Remove features that are not available
        all_features = {
            name: feature
            for name, feature in all_features.items() if feature is not None
        }
        return all_features

    @sw.decorate
    def _merge(self):
        if len(self._feature_data)==0:
            raise RuntimeError('No data to save.')
        keys = self._feature_data[0].keys()
        # Merge all steps into a single array dictionary
        # NOTE: this assumes that all dictionaries have the same keys
        merged_dict = {
            k: np.array([d[k] for d in self._feature_data])
            for k in keys
        }
        return merged_dict

    @sw.decorate
    def _post_process(self, merged_dict):
        merged_dict = merged_dict.copy()
        if 'dead_units' in merged_dict:
            merged_dict['next_dead'] = self._compute_next_dead(merged_dict['dead_units'])
        return merged_dict

    @sw.decorate
    def _compute_next_dead(self, dead_units):
        next_dead_arr = np.zeros(len(dead_units))
        next_dead = 0
        for i, dead in enumerate(np.flip(dead_units)):
            if dead > 0:
                next_dead = 0
            else:
                next_dead += 1
            next_dead_arr[i] = next_dead
        return np.flip(next_dead_arr)

    @sw.decorate
    def _post_process_raw(self):
        # merging from list of images to ndarray of images
        self._merge_and_clip_raw()
        # summarizing over time
        self._summarize_raw_over_time()
        return None

    def _summarize_raw_over_time(self):
        time_dim = 0
        # summarizing non-tabular features (e.g., hyperspectral and  map state):
        for name in [key for key in self._raw_data_dict if 'tabular' not in key]:
            data = self._raw_data_dict[name]  # Shape (time, ...)
            assert data.shape[0] % self.summarize_step == 0, f'{name} is not divisible by summarize step.'

            n_total_frames = data.shape[time_dim]
            n_per_window = self.summarize_step
            n_windows = n_total_frames // n_per_window

            weight_shape = ([1] * data.ndim)
            weight_shape[time_dim] = n_per_window  # Set it explicitly to catch errors
            weights = (np.arange(n_per_window) + 1).reshape(weight_shape).astype(np.uint8)

            assert data.dtype == bool, f'The hyperspectral/map data must be of type bool. Got type {data.dtype}'
            summarized_data = (np_sparse if isinstance(data, np_sparse.COO) else np).stack([
                (data[ei*n_per_window:(ei+1)*n_per_window] * weights).max(axis=time_dim).astype(np.uint8)
                for ei in range(n_windows)
            ])

            # overwriting old data with summarized data
            self._raw_data_dict[name] = summarized_data

        # summarizing tabular features
        for name in [key for key in self._raw_data_dict if 'tabular' in key]:
            data = self._raw_data_dict[name]  # Shape (time, ...)
            assert data.shape[0] % self.summarize_step == 0, f'{name} is not divisible by summarize step.'
            summarized_data = np.stack([
                data[ei*n_per_window:(ei+1)*n_per_window].mean(axis=time_dim).astype(np.int32)
                for ei in range(n_windows)
            ])
            current_data = np.stack([
                data[(ei+1)*n_per_window - 1].astype(np.int32)
                for ei in range(n_windows)
            ])
            self._raw_data_dict[name] = current_data  # Overwrite with current tabular
            # Add average tabular as another entry
            # NOTE: this is okay since we already made a list of keys above
            self._raw_data_dict[name+'_average'] = summarized_data
        return None    

    @sw.decorate
    def _merge_and_clip_raw(self):
        """Stacks list of images into ndarrays or np_sparse arrays, and trims them so they are
        divisible by self.summarize_mul -- the trimming happens at the beginning of the array"""
        n_total_frames = len(self._raw_data_dict['hyperspectral'])
        if n_total_frames < self.obs_counter:
            warn(f'WARNING: Extracted {n_total_frames} but the replay_info has {self.obs_counter}')
        frame_skip_point = (n_total_frames // self.summarize_step) * self.summarize_step
        clipped_slice = slice(-frame_skip_point, None)
         
        for name in self._raw_data_dict:
            if isinstance(self._raw_data_dict[name][0], np_sparse.COO):
                self._raw_data_dict[name] = np_sparse.stack(self._raw_data_dict[name][clipped_slice])
            else:
                self._raw_data_dict[name] = np.stack(self._raw_data_dict[name][clipped_slice])
        self._metadata = self._metadata[clipped_slice]
        return None

    @sw.decorate
    def _extract_just_player_raw_features(self, obs, player_id):
        """This extracts the visibility, creep, and tabular data only for player `player_id`, and then appends
        it to the running section for player_id in the `self._raw_data_dict` dictionary
        Note: This is done as a separate method from the `extract_raw_features` function in order
              to not corrupt the stopwatch recording for the main replay extraction"""
        raw_data = obs.observation.raw_data
        # since raw creep and raw vis are in the original map size, we need to crop to playable area
        raw_creep = self.extract_crop_and_resize_map_data(raw_data.map_state.creep).astype(bool)
        raw_visibility = self.extract_crop_and_resize_map_data(raw_data.map_state.visibility)
        
        raw_is_visible = raw_visibility == 2
        raw_is_seen = np.isin(raw_visibility, [1,2])

        raw_tabular = self.extract_tabular(obs.observation.player_common)

        # add the extracted raw data to the running raw dictionary
        self._raw_data_dict[f'player_{player_id}_map_state'].append(np.stack([raw_is_visible, raw_is_seen, raw_creep]))
        self._raw_data_dict[f'player_{player_id}_tabular'].append(raw_tabular)
        return None

    @sw.decorate
    def extract_raw_features(self, obs):
        """This creates a hyperspectral image from raw data and then appends
        it to the running `self._raw_data_dict` dictionary"""
        raw_data = obs.observation.raw_data

        hyperspectral_image = self.make_hyperspectral_image(raw_data.units)

        # add the extracted raw data to the running raw dictionary
        self._raw_data_dict['hyperspectral'].append(hyperspectral_image)
        return None

    @sw.decorate
    def extract_tabular(self, player_common):
        return np.array([getattr(player_common, k) for k in PLAYER_COMMON_KEYS])

    @sw.decorate
    def make_hyperspectral_image(self, raw_units, just_return_indices_values=False):
        image_indices = []  # the adjusted locations of each unit for the hyperspectral image
        image_values = []   # for now this will just be 1 if a unit is present

        # finding hyperspectral positions 
        for unit in raw_units:
             # NOTE: to take the size of the unit into account during converting, set `size=unit.radius` below
            unit_position = self.convert_raw_unit_location_to_hyperspectral(unit.pos, size=None)

            # getting the channel which to place the unit
            if unit.owner == 16:
                # unit is of neutral type
                unbiased_channel_idx = self.neutral_id_to_channel.get(unit.unit_type)
                assert unbiased_channel_idx is None or unbiased_channel_idx < len(neutral_ids)
            else:
                # unit is of nonneutral type
                unbiased_channel_idx = self.nonneutral_ids_to_channel.get(unit.unit_type)
                assert unbiased_channel_idx is None or unbiased_channel_idx < len(nonneutral_ids)

            if unbiased_channel_idx is None:
                # there are some unit_type ids that are not in the static_data, so we use the first index
                unbiased_channel_idx = UNKNOWN_CHANNEL
            if unit.owner not in self.player_type_to_channel_offsets:
                if not unit.is_blip:
                    warn(f'SkipWarning: Skipping unit because unit owner = {unit.owner}'
                         f'but is_blip = {unit.is_blip}, full UNIT information:\n{unit}')
                continue # Skip
            channel_idx = unbiased_channel_idx + self.player_type_to_channel_offsets[unit.owner]
            
            unit_value = 1  # in case we ever want to change this

            if isinstance(unit_position[0], (list, )):
                # the unit size was larger than 1, so it spans multiple pixels
                for sub_location in unit_position:
                    image_indices.append([channel_idx, sub_location[0], sub_location[1]])
                    image_values.append(unit_value)  
            else:
                # the size was one, so no need to loop over locations
                image_indices.append([channel_idx, unit_position[0], unit_position[1]])
                image_values.append(unit_value)  

        # making of the sparse ndarray
        hyperspectral_image = np_sparse.COO(np.array(image_indices).T, image_values,
                                                     shape=self.hyperspectral_image_size).astype(bool)
        
        if just_return_indices_values:
            return hyperspectral_image.indices(), hyperspectral_image.values()
        else:
            return hyperspectral_image

    @sw.decorate
    def convert_raw_unit_location_to_hyperspectral(self, raw_unit_location, size=None):
        # assert raw_unit_location.x < raw_map_size.x and raw_unit_location.y < raw_map_size.y, \
                # "Raw unit location must be smaller than raw_map_size"

        # offsetting raw unit location due to cropping to playable area
        raw_unit_location = Point(raw_unit_location.x - self.playable_area_0.x,
                                  raw_unit_location.y - self.playable_area_0.y)

        if raw_unit_location.x >= self.raw_map_size.x:
            message = f'Raw unit location {raw_unit_location.x, raw_unit_location.y} is greater than' + \
                    f'map size {self.raw_map_size.x, self.raw_map_size.y}, so clipping to below map size.'
            warn(message)
            raw_unit_location = Point(self.raw_map_size.x - 1, raw_unit_location.y)
        if raw_unit_location.y >= self.raw_map_size.y:
            message = f'Raw unit location {raw_unit_location.x, raw_unit_location.y} is greater than' + \
                    f'map size {self.raw_map_size.x, self.raw_map_size.y}, so clipping to below map size.'
            warn(message)
            raw_unit_location = Point(raw_unit_location.x, self.raw_map_size.y - 1)
        
        # note: int() always truncates float  (it does not round)
        new_position = (int(raw_unit_location.x*self.raw_width_to_hyper_multiplier),
                        int(raw_unit_location.y*self.raw_height_to_hyper_multiplier))

        if size is None:
            return new_position
        
        else:
            new_size = int(np.ceil((max(self.raw_image_size) / self.raw_map_max_dim) * size))
            if new_size == 0:
                # TODO: handle this better, e.g. supporting a return of None
                # for now, this is a very rare case, and is low priority
                new_size = 1

            if new_size == 1:
                # no need to account for size since it is already 1 pixel wide
                return new_position
            
            # we need to pad position to size, note: we pad left first and then right.
            # so the padding will be happen asymmetrically if new_size is even
            # i.e. if new_position = (4,5) and size=2, new_positions = ((3,4), (3,5), (4,4), (4,5))
            if new_size % 2 == 0:
                range_offset = 0
            else:
                range_offset = 1
            new_positions = []
            x_range = list(range(new_position[0]-new_size//2, range_offset+new_position[0]+new_size//2))
            y_range = list(range(new_position[1]-new_size//2, range_offset+new_position[1]+new_size//2))
            for x_coord in x_range:
                for y_coord in y_range:
                    if (0 <= x_coord < self.raw_map_size.x) and (0 <= y_coord < self.raw_map_size.y):
                        new_positions.append([x_coord, y_coord])
            return new_positions

    @sw.decorate
    def extract_crop_and_resize_map_data(self, map_state, return_kind='numpy', pad_value=None):
        map_data = self.extract_raw_map_data_from_buffer(map_state)
        map_data = self._crop_raw_image_to_playable_area(map_data, pad_value=pad_value)
        map_data = self._resize_raw_image(map_data, kind=return_kind)
        return map_data

    @sw.decorate
    def _crop_raw_image_to_playable_area(self, image, pad_value=None):
        assert image.ndim == 2
        image = np.fliplr(image) # First flip the x direction to match unit positions
        # Now crop to playable area
        image = image[self.playable_area_0.x:self.playable_area_1.x,
                      self.playable_area_0.y:self.playable_area_1.y]
        # Now pad to make square since needs to be square
        assert self.raw_image_size[0] == self.raw_image_size[1], 'We assume the raw image size is a square'
        nr, nc = image.shape
        pad = int(np.abs(nr - nc))
        if pad_value is None:
            pad_value = image.min()
        if nr > nc:  # Rows > Cols so add columns
            image = np.pad(image, ((0, 0), (0, pad)), constant_values=pad_value)
        elif nc < nr:  # Cols > Rows so pad rows (i.e., x)
            image = np.pad(image, ((0, pad), (0, 0)), constant_values=pad_value)
        return image

    @sw.decorate
    def _resize_raw_image(self, image, wanted_size=None, kind='numpy', do_anti_aliasing=False):
        if wanted_size is None:
            wanted_size = self.raw_image_size

        if do_anti_aliasing:
            resized_image = resize(image, wanted_size)
        else:
            # some raw_images must keep specific value (e.g., for visibility 0=hidden, 1=seen, 2=visible, else=???)
            resized_image = resize(image, wanted_size, order=0, anti_aliasing=False)

        if kind == 'torch' or kind == 'pytorch':
           resized_image = torch.tensor(resized_image)
        return resized_image


    # # @sw.decorate
    # @staticmethod
    # def collapse_to_last_time_recording(frame_stack, time_dim=0):
    #     """Returns a frame_stack which shows the last time a unit was see at each coordinate"""
    #     assert frame_stack.dtype == bool, f'The frame stack must be of type bool. Got type {frame_stack.dtype}'
    #     weight_shape = ([1]*frame_stack.ndim)
    #     weight_shape[time_dim] = -1
    #     weights = np.arange(1, frame_stack.shape[time_dim]+1).reshape(weight_shape)
    #     weighted_frame_stack = frame_stack * weights
    #     return weighted_frame_stack.max(time_dim).astype(np.uint8)

    @staticmethod
    @sw.decorate
    def extract_raw_map_data_from_buffer(image_data_object):
        """This takes in a map_state from raw (e.g. raw.map_state.visibility) and 
        returns the unpacked map data.
        NOTE: this should be a specific ImageData type (e.g. raw.map_state.creep), 
            not directly a MapState object"""
        extracted_data = np.frombuffer(image_data_object.data,
                                       dtype=features.Feature.dtypes[image_data_object.bits_per_pixel])
        if extracted_data.shape[0] != (image_data_object.size.x * image_data_object.size.y):
            # This could happen if the correct length isn't a multiple of 8, leading
            # to some padding bits at the end of the string which are incorrectly
            # interpreted as data.
            extracted_data = extracted_data[:image_data_object.size.x * image_data_object.size.y]
        return extracted_data.reshape(image_data_object.size.x, image_data_object.size.y, order='F')
