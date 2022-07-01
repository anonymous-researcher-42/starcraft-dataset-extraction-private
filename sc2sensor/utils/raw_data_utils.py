from pathlib import Path
from warnings import warn

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize

from pysc2.lib import static_data 
from pysc2.lib import features

from sc2sensor.utils.image_utils import imshow

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

def make_color_context_battle_frame(player_relative, dtype=float, multiplyer=1):
    """Colorizing battle frame based off of friendly, hostile, or neutral
    player_relative: Which units are friendly vs hostile. """
    from pysc2.lib import features
    PLAYER_SELF = features.PlayerRelative.SELF
    PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals 
    PLAYER_ENEMY = features.PlayerRelative.ENEMY
    
    blue_team = player_relative== PLAYER_SELF
    neutral = player_relative == PLAYER_NEUTRAL
    red_team = player_relative == PLAYER_ENEMY
    return np.stack([red_team, neutral, blue_team]).astype(dtype) * multiplyer

def create_RBG_image_from_raw_units(raw_frame_data, image_size=(256, 256), use_size=True):
    # param: raw_frame_data == obs.observation.raw_data
    image = np.zeros((image_size[0], image_size[1]))  # 1 channel for now (will be split into 3 at the end)
    raw_map_size = raw_frame_data.map_state.visibility.size

    for raw_unit in raw_frame_data.units:
        if use_size:
            unit_location = convert_raw_unit_location_to_hyperspectral(raw_unit.pos, raw_map_size,
                                                                                image_size, raw_unit.radius)
        else:
            unit_location = convert_raw_unit_location_to_hyperspectral(raw_unit.pos, raw_map_size,
                                                                    image_size)
        if isinstance(unit_location[0], (list, )):
            for sub_location in unit_location:
                image[tuple(sub_location)] = raw_unit.alliance
        else:
            # the size was one, so no need to loop over locations
            image[unit_location] = raw_unit.alliance

    colored_image = make_color_context_battle_frame(image)
    return colored_image

def convert_raw_unit_location_to_hyperspectral(raw_unit_location, raw_map_size,
                                               new_image_size=(256, 256), size=None):
    # assert raw_unit_location.x < raw_map_size.x and raw_unit_location.y < raw_map_size.y, \
            # "Raw unit location must be smaller than raw_map_size"
    if raw_unit_location.x >= raw_map_size.x:
        message = f'Raw unit location {raw_unit_location.x, raw_unit_location.y} is greater than' + \
                  f'map size {raw_map_size}, so clipping to below map size.'
        warn(message)
        raw_unit_location.x = raw_map_size.x - 1
    if raw_unit_location.y >= raw_map_size.y:
        message = f'Raw unit location {raw_unit_location.x, raw_unit_location.y} is greater than' + \
                  f'map size {raw_map_size}, so clipping to below map size.'
        warn(message)
        raw_unit_location.y = raw_map_size.y - 1

    max_dim = max(raw_map_size.x, raw_map_size.y)
    
    width_multiplier = new_image_size[0] / max_dim
    height_multiplier = new_image_size[1] / max_dim
    # note: int() always truncates float  (it does not round)
    new_position = int(raw_unit_location.x*width_multiplier), int(raw_unit_location.y*height_multiplier)

    if size is None:
        return new_position
    
    else:
        new_size = int(np.ceil((max(new_image_size) / max_dim) * size))
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
                # making sure coords are in the new_image range
                if (0 <= x_coord < new_image_size[0]) and (0 <= y_coord < new_image_size[1]):
                    new_positions.append([x_coord, y_coord])
        return new_positions
def make_hyperspectral_image(raw_units, raw_map_size,
                             hyperspectral_size=(256, 256),
                             return_indices_values=True):

    # setting up idx mappings
    all_unit_types = static_data.UNIT_TYPES.copy()
    all_unit_types.insert(0, -1)  # inserts a -1 unit type which we define to be "other"
    unit_type_to_channel = {unit_type: channel_idx for channel_idx, unit_type in enumerate(all_unit_types)}
    # each player gets their own set of channels for each unit type, so we need to account for the owner of each unit
    player_type_to_channel_offsets = {1:  0,                      # 1  := self player
                                      2:  len(all_unit_types),    # 2  := enemy players
                                      16: 2*len(all_unit_types),  # 16 := neutral players (i.e. materials)
                                     }  # used to find the right channel index based off player_type and unit_type

    # we only support two players (+ neutral), so confirming this is the case:
    player_types = [unit.owner for unit in raw_units]
    np.testing.assert_equal(np.unique(player_types), [1, 2, 16], err_msg='Only two players are supported.')

    n_hyperspectral_channels = len(player_type_to_channel_offsets) * len(unit_type_to_channel)

    image_indices = []  # the adjusted locations of each unit for the hyperspectral image
    image_values = []   # for now this will just be 1 if a unit is present

    # finding hyperspectral positions 
    for unit in raw_units:
        unit_position = convert_raw_unit_location_to_hyperspectral(unit.pos, raw_map_size,
                                                              hyperspectral_size, unit.radius)
        unbiased_channel_idx = unit_type_to_channel.get(unit.unit_type)
        if unbiased_channel_idx is None:
            # there are some unit_type ids that are not in the static_data, so we use -1 to represent these
            unbiased_channel_idx = -1
        channel_idx = unbiased_channel_idx + player_type_to_channel_offsets[unit.owner]

        unit_value = 1  # in case we ever want to change this
        if isinstance(unit_position[0], (list, )):
            for sub_location in unit_position:
                image_indices.append([channel_idx, sub_location[0], sub_location[1]])
                image_values.append(unit_value)  
        else:
            # the size was one, so no need to loop over locations
            image_indices.append([channel_idx, unit_position[0], unit_position[1]])
            image_values.append(unit_value)  

    # making of the sparse tensor
    hyperspectral_image = torch.sparse_coo_tensor(torch.tensor(image_indices).T, image_values,
                                                  size=(n_hyperspectral_channels, *hyperspectral_size))
    # summing any duplicate entries (i.e. getting coordinate-wise counts)
    hyperspectral_image = hyperspectral_image.coalesce()
    
    # TODO: rotate the sparse image?
    # # need to do with the indices since sparse_coo images don't support a rot90 
    
    if not return_indices_values:
        return hyperspectral_image
    else:
        return hyperspectral_image, image_indices, image_values

def make_sparse_rbg_and_player_relative_comparison_plots(obs, rotate_non_player_relative=True,
                                                         hyperspectral_size=(64,64),
                                                         save_filename=None,
                                                         show=True):
    # plotting the three plots for comparison
    # for some reason we need to rotate the images created from raw_units by 90 in order to match the player relative

    fig, axes = plt.subplots(1,3, figsize=(15,5))

    # making hyperspectral_image
    raw = obs.observation.raw_data
    hyperspectral_image, image_indices, image_values = make_hyperspectral_image(raw.units,
                                                                             raw.map_state.visibility.size,
                                                                             hyperspectral_size=hyperspectral_size,
                                                                             return_indices_values=True)
    # collapsing the hyperspectral_image to 1D
    one_d_projected_hyperspectral = np.zeros((hyperspectral_size))
    for idxs in image_indices:
        one_d_projected_hyperspectral += hyperspectral_image[idxs[0]].to_dense().numpy()
    one_d_projected_hyperspectral = (one_d_projected_hyperspectral != 0) + 0  # makes the image binary
    if rotate_non_player_relative:
        one_d_projected_hyperspectral = np.rot90(one_d_projected_hyperspectral, k=1)
    imshow(one_d_projected_hyperspectral, axes[0], title='Sparse tensor --> 1 channel')

    rgb_from_raw = create_RBG_image_from_raw_units(raw, image_size=hyperspectral_size)
    if rotate_non_player_relative:
        rgb_from_raw = np.rot90(rgb_from_raw, k=1, axes=(1,2))
    imshow(rgb_from_raw, axes[1], title='RBG (player relative) built from raw unit')

    player_relative = features.MINIMAP_FEATURES.player_relative.unpack(obs.observation)
    imshow(make_color_context_battle_frame(player_relative), axes[2], title='Player relative plot')

    if save_filename is not None:
        plt.savefig(save_filename)
    if show:    
        plt.show()

def resize_raw_image(image, wanted_size, kind='numpy', do_anti_aliasing=False):
    if do_anti_aliasing:
        resized_image = resize(image, wanted_size)
    else:
        # some raw_images must keep specific value (e.g., for visibility 0=hidden, 1=seen, 2=visible, else=???)
        resized_image = resize(image, wanted_size, order=0, anti_aliasing=False)

    if kind == 'torch' or kind == 'pytorch':
        resized_image = torch.tensor(resized_image)
    return resized_image



# TODO: move below to a better location
def make_weights_constant(T):
    return torch.ones(T) / T
def make_weights_exponential(T, r):
    base = r / torch.exp(torch.tensor(T-1))
    # r = a^(T-1)/a^(0) = a^(T-1) -- Solve for a
    # a = r^(1/T-1)
    base = r**(1/(T-1))
    weights = torch.pow(base, torch.arange(T) - T) # The - T is for numerical stability
    weights = torch.flip(weights, (0,))
    #print('base=', base)
    #print('r=', r, 'ratio=', weights[0] / weights[-1], )
    assert torch.isclose(weights[0] / weights[-1], torch.tensor([r]), rtol=1e-3), 'expected ratio between smallest and largest is not met'
    return weights / torch.sum(weights)
def make_weights_linear(T, r):
    # r = (0+a)/(T-1+a) # Solve for a
    a = (r / (1 - r)) * (T - 1)
    weights = torch.arange(T) + a
    actual_ratio = weights[0] / weights[-1]
    #print('a=', a)
    #print('r=', r, 'ratio=', weights[0] / weights[-1])
    assert torch.isclose(weights[0] / weights[-1], torch.tensor([r])), 'expected ratio between smallest and largest is not met'
    return weights / torch.sum(weights)
