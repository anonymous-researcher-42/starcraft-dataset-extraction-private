import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.unit_type_data import NONNEUTRAL_IDS, NEUTRAL_IDS


class _CheckLogitModule(nn.Module):
    def __init__(self, is_weight_logit=False):
        super().__init__()
        self._is_weight_logit = is_weight_logit

    def _check_logit(self, x):
        if self._is_weight_logit:
            clip_eps = torch.finfo(x.dtype).eps
            return torch.logit(torch.clamp(x, min=clip_eps, max=1-clip_eps))
        else:
            return x

    @property
    def is_weight_logit(self):
        return self._is_weight_logit


class StarCraftUnitEmbedding(_CheckLogitModule):
    def __init__(self, is_neutral=False, embed_size=1, **kwargs):
        super().__init__(**kwargs)
        if is_neutral:
            n_input = len(NEUTRAL_IDS)
        else:
            n_input = len(NONNEUTRAL_IDS)
        self.embed_param = nn.Parameter(self._check_logit(torch.ones((n_input, embed_size))))
        
    def forward(self, x):
        '''x should be unit_ids tensor and this extracts the value for that tensor'''
        weights = self.embed_param
        if self._is_weight_logit:
            weights = torch.sigmoid(weights)
        return F.embedding(x, weights)


class StarCraftToImageReducer(_CheckLogitModule):
    def __init__(self, init_player_dense_weight=(1,), init_neutral_dense_weight=(0.2,), **kwargs):
        super().__init__(**kwargs)
        # Embedding is just to extract values from unit_type_ids
        #  This is similar to weighting each channel
        embed_size = 1
        # The player unit embedding is the same for both players
        self.player_unit_embed = StarCraftUnitEmbedding(is_neutral=False, embed_size=embed_size)
        self.neutral_unit_embed = StarCraftUnitEmbedding(is_neutral=True, embed_size=embed_size)
        
        # Unit and creep (note that creep is like a special unit type)
        self.player_dense_weight = nn.Parameter(
            self._check_logit(torch.tensor(init_player_dense_weight).float()))
        # Map data including neutral_unit, pathing_grid, terrain_height, and placement_grid
        self.neutral_dense_weight = nn.Parameter(
            self._check_logit(torch.tensor(init_neutral_dense_weight).float()))
        
    def forward(self, d):
        # Check if batch dimension exists
        channel_setup = [
            ('player_2', self.player_unit_embed), # Red - Enemy
            ('neutral', self.neutral_unit_embed),  # Green - Neutral
            ('player_1', self.player_unit_embed),  # Blue - Player
        ]
        # Embed units into single layers
        channels = [
            self._channel_reduce(d[f'{pre}_unit_ids'], d[f'{pre}_unit_values'], embed)
            for pre, embed in channel_setup
        ]
        # Embed other information into each layer
        channels = [
            self._dense_channel_reduce(ch, pre, d)
            for ch, (pre, _) in zip(channels, channel_setup)
        ]
        x = torch.cat(channels, dim=1)  # Concatenate on channel dimension
        return x
                                  
    def _dense_channel_reduce(self, unit, pre, d):
        if pre.startswith('player'):
            w = self.player_dense_weight # (C,)
            # map state is [is_visible, is_seen, creep]
            # Ignoring creep for now 
            #creep = d[f'{pre}_map_state'][:,2,:,:] # (B, H, W)
            #dense = [creep]
            dense = []
        else:
            w = self.neutral_dense_weight # (C,)
            # All are (B, H, W)
            # Ignore terrain features for now
            #dense = [d[f'pathing_grid'], d[f'terrain_height'], d[f'placement_grid']]
            dense = []
        # Unsqueeze so that we can cat
        dense = [t.unsqueeze(1) for t in dense]  # (B, 1, H, W)
        # Combine all dense mats with unit dense
        x = torch.cat([unit, *dense], axis=1) # (B, 1+D, H, W) where D = len(dense)
        
        # Get weight and apply 
        if self._is_weight_logit:
            w = torch.sigmoid(w)
        x = x * w.reshape(-1, 1, 1) # (B, 1+D, H, W)
        
        # Reduce via max
        return x.amax(axis=-3, keepdim=True) # (B, 1, H, W)
        
    def _channel_reduce(self, x, w, embed):
        '''x should be a LongTensor of shape (B, C', W, H) where
        B is the batch dimension, C' is a variable number of channels
        equal to the max overlap for this batch, and W and H are the 
        width and height of the image.
        '''
        # Extract learned weights via embedding
        x = embed(x)  # (B, C', W, H, E) where E is the embed_size
        # Multiply learned weights by static timestamp values
        #  (Need to add 1 dimension to broadcast since weight is (B, C', W, H))
        x = x * w.unsqueeze(dim=-1)  # (B, C', W, H, E) 
        # Summarize across all channels by reducing over C'
        #  (Reduce over C' (variable) should be sum or max (mean is not good since C' is variable))
        #  (Use -4 in case the batch dimensions are not a single dimension)
        x = x.amax(axis=-4)  # (B, W, H, E)
        # Permute to put new embedding dimension as channel dimension
        x = x.permute((0, 3, 1, 2)) # (B, E, W, H)
        # Note that no matter what C' was, it is now a fixed E number of channels
        return x


# OLD VERSION: Retained for backwards compatability
from .utils.unit_type_data import _OLD2_NONNEUTRAL_IDS, _OLD2_NEUTRAL_IDS

class _OLD_StarCraftUnitEmbedding(_CheckLogitModule):
    def __init__(self, is_neutral=False, embed_size=1, **kwargs):
        super().__init__(**kwargs)
        if is_neutral:
            # Start at a smaller value since not as important
            self.embed_param = nn.Parameter(self._check_logit(0.2 * torch.ones((len(_OLD2_NEUTRAL_IDS), embed_size))))
        else:
            self.embed_param = nn.Parameter(self._check_logit(torch.ones((len(_OLD2_NONNEUTRAL_IDS), embed_size))))
        # To handle the "no unit at this location" value which is set to -1 currently
        self.embed_no_unit_param = nn.Parameter(self._check_logit(0 * torch.ones_like(self.embed_param[0,:].view(1,-1))))
        
    def forward(self, x):
        '''x should be unit_ids tensor and this extracts the value for that tensor'''
        # Because nothing is coded is ID=-1, we will append a the no_unit parameter
        #  and increment x by 1 before applying the embedding layer.
        weights = torch.vstack((self.embed_no_unit_param, self.embed_param))
        if self._is_weight_logit:
            weights = torch.sigmoid(weights)
        return F.embedding(x + 1, weights)

