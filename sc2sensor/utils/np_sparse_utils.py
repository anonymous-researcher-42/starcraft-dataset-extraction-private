import sparse as np_sparse
import numpy as np
import torch
from warnings import warn

def diff_over_time(frame_stack, time_axis=0, normalize=True):
    if isinstance(frame_stack, (np_sparse.COO, )):
        diffed_sparse = []
        for time_idx in range(frame_stack.shape[0]-1):
            diffed_sparse.append(frame_stack[time_idx + 1] - frame_stack[time_idx])
        diffed_sparse = np.abs(np_sparse.stack(diffed_sparse)).mean(axis=time_axis)
        if normalize:
            diffed_sparse = (diffed_sparse - diffed_sparse.min()) / (diffed_sparse.max() - diffed_sparse.min() + 1e-8)
        return diffed_sparse
    else:
        # input must be a ndarray
        averaged_diff = np.abs(np.diff(frame_stack, axis=time_axis)).mean(axis=time_axis)
        if normalize:
            averaged_diff = (averaged_diff - averaged_diff.min()) / (averaged_diff.max() - averaged_diff.min() + 1e-8)
        return averaged_diff

# max over time axis:
# weighted max over time axis:
def max_over_time(frame_stack, weights=None, time_dim=0):
    """ A max function which can handle sparse or dense tensors tensors and
    returns a tensor which only holds the max value for each point across the time axis
    tensor_to_max_over.shape = (n_time, dim1, ...,)"""
    frame_stack = make_np_or_np_sparse(frame_stack)
    if weights is not None:
        weights = make_np_or_np_sparse(weights)
        weights = weights.astype(float)
        assert weights.ndim == 1 and weights.shape[0] == frame_stack.shape[time_dim], \
                'weights.shape should equal the size of the time axis'
        assert np.all(0 <= weights ), 'The weight vector should be positive'
        weights = weights / weights.sum()  # normalizing to probability vector and scaling by time_dim_size
        # multiplying weights along the time axis
        # setting up weights so it can be broadcasted along the time_dim
        weight_reshape = ([1]*frame_stack.ndim)
        weight_reshape[time_dim] = -1
        frame_stack = frame_stack * weights.reshape(weight_reshape) * weights.shape[0]
    return frame_stack.max(time_dim)


# averaging
def average_over_time(frame_stack, weights=None, time_dim=0):
    """ Averages a tensor across the time axis, where `frame_stack` can be dense or sparse.
    If `weights` is not `None`,   weights.shape := (n_time)"""
    frame_stack = make_np_or_np_sparse(frame_stack)
    
    time_dim_size = frame_stack.shape[time_dim]
    
    is_sparse = isinstance(frame_stack, (np_sparse.COO, ))

    assert 0 <= time_dim < frame_stack.ndim, \
            'time_dim must be a number in [0, frame_stack.ndim)'
        
    # the tensor must be floating point in order for averaging to work properly
    frame_stack = frame_stack.astype(float)
    
        
    if weights is None:
        time_dim_size = frame_stack.shape[time_dim]
        return frame_stack.sum(time_dim) / time_dim_size
    else:
        weights = make_np_or_np_sparse(weights)
        weights = weights.astype(float)
        assert weights.ndim == 1 and weights.shape[0] == frame_stack.shape[time_dim], \
                'weights.shape should equal the size of the time axis'
        assert np.all(0 <= weights ), 'The weight vector should be positive'
        weights = weights / weights.sum()  # normalizing to probability vector and scaling by time_dim_size
        # multiplying weights along the time axis
        # setting up weights so it can be broadcasted along the time_dim
        weight_reshape = ([1]*frame_stack.ndim)
        weight_reshape[time_dim] = -1
        frame_stack = frame_stack * weights.reshape(weight_reshape)
        return frame_stack.sum(axis=time_dim)


# def sparse_tensor_to_sparse_matrix(sparse_tensor, dim_to_keep=1):
#     """This essentially acts as a general move + 2d reshaping of a sparse tensor to a sparse matrix where
#     dim_to_keep is the dimension to hold constant and all others are flattened. 
#     It is equivalent to the dense operation: `dense_tensor.moveaxis(dim_to_keep, 0).reshape(shape[dim_to_keep], -1).`
#     The `sparse_tensor.ndim` must be greater or equal to 2, and the dim_to_keep must be \in [0, sparse_tensor.ndim]"""
#     # In: sparse_tensor.shape = [dim_0, dim_1, ..., dim_k]
#     # Out: sparse_matrix.shape = [dim_to_keep, product(dims)/dim_to_keep]
#     assert 0 <= dim_to_keep < sparse_tensor.ndim, 'dim_to_keep must be a number in [0, sparse_tensor.ndim)'
    
#     sparse_tensor = sparse_tensor.coalesce()
    
#     # first we permute the sparse tensor indices so that the dim_to_keep becomes the first dim
#     perm = list(range(sparse_tensor.ndim))
#     perm.pop(dim_to_keep)
#     perm.insert(0, dim_to_keep)
    
#     indices = sparse_tensor.indices()
#     perm_indices = indices[perm, :]
    
#     perm_shape = torch.tensor(sparse_tensor.shape)[perm]
#     return_matrix_shape = (perm_shape[0], torch.prod(perm_shape[1:]))
    
#     # setting up strides
#     dim_strides = torch.tensor([torch.prod(perm_shape[i:]).item() for i in range(2,sparse_tensor.ndim)] + [1])
#     # raveling the indices from perm_dims[1:]
#     matrix_col_indices = torch.sum(perm_indices[1:].T * dim_strides, axis=1)
    
#     new_indices = torch.stack((perm_indices[0], matrix_col_indices))

#     return torch.sparse_coo_tensor(new_indices, sparse_tensor.values(), size=return_matrix_shape).coalesce()

# def sparse_1x1_convolution(sparse_tensor, weight_mat):
#     # In: sparse_tensor.shape = [n_batch, n_channels, h, w]
#     #     weight_mat.shape = [n_channels_in, n_channels_out]
    
#     assert weight_mat.ndim == 2, 'weight_mat must has shape [n_channels, n_batch*h*w]'
#     assert sparse_tensor.shape[1] == weight_mat.shape[0]
#     # flattening sparse_tensor to sparse_matrix
#     sparse_matrix = sparse_tensor_to_sparse_matrix(sparse_tensor, dim_to_keep=1)
#     # convolving sparse_matrix
#     convolved_matrix = sparse_matrix.t().mm(weight_mat).t()
#     # reshaping convolved_matrix back to the original shape (with n_channels_out)
#     prepermute_output_shape = (weight_mat.shape[1], sparse_tensor.shape[0],
#                                sparse_tensor.shape[2], sparse_tensor.shape[3])
#     convolved_matrix = convolved_matrix.reshape(prepermute_output_shape).moveaxis(1,0)
    
#     return convolved_matrix


def create_random_sparse_tensor(size, n_values, arange=False):
    """A helper function for quickly creating sparse tensor of a specified size"""
    indices = np.vstack([ np.random.randint(0, s, size=(n_values, )) for s in size])
    if arange:
        values = np.arange(n_values)
    else:
        values = np.randn(n_values)
    
    return np_sparse.COO(indices, values, size)

def to_np_sparse(sparse_tensor_or_dict):
    if isinstance (sparse_tensor_or_dict, (dict, )):
        return np_sparse.COO(sparse_tensor_or_dict['indices'], sparse_tensor_or_dict['values'],
                             sparse_tensor_or_dict['shape'])
    sparse_tensor = sparse_tensor_or_dict.coalesce()
    inds, vals = sparse_tensor_or_dict.indices(), sparse_tensor_or_dict.values()
    shape = sparse_tensor_or_dict.shape
    return np_sparse.COO(inds.numpy(), vals.numpy(), tuple(shape))

def make_np_or_np_sparse(x):
    if isinstance(x, (np_sparse.COO, )):
        return x
        
    if isinstance(x, (torch.Tensor, )):
        if x.is_sparse:
            warn('The input is a sparse tensor.' + \
                  'We will try to convert it to an np_sparse format, but this should be avoided.')
            # return a np_sparse.COO tensor
            x = x.coalesce()
            return np_sparse.COO(x.indices(), x.values(), tuple(x.shape))
        else:
            return x.numpy()
    return np.array(x)
    # assert isinstance(x, (np.ndarray, np_sparse.COO, )), f'Input must be of np type. Received {type(x)} type.'

def _assert_np_sparse(x):
    assert isinstance(x, (np_sparse.COO, )), f'Input must be of sparse (np_sparse) type. Received {type(x)} type.'