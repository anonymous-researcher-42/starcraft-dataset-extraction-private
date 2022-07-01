import torch

# max over time axis:
def max_over_time(tensor_to_max_over):
    """ A max function which can handle sparse or dense tensors tensors and
    returns a tensor which only holds the max value for each point across the time axis
    tensor_to_max_over.shape = (n_time, dim1, ...,)"""
    time_dim = 0  # for now we'll only support the time_dim being the first dimension
        
    if tensor_to_max_over.is_sparse:
        # this is *very* inefficient, but works. #TODO: Update with faster code
        return_shape = list(tensor_to_max_over.shape)
        return_shape.pop(time_dim)
        time_maxes = tensor_to_max_over[0].to_dense()
        for time_idx in range(1, tensor_to_max_over.shape[time_dim]):
            time_maxes = torch.max(torch.stack((time_maxes,
                                                tensor_to_max_over[time_idx, ...].to_dense())), axis=0)[0]
        return time_maxes.to_sparse_coo()
    else:
        return tensor_to_max_over.max(dim=time_dim)[0]


# averaging
def average_over_time(tensor_to_average, weights=None, time_dim=0):
    """ Averages a tensor across the time axis, where `tensor_to_average` can be dense or sparse.
    If `weights` is not `None`,   weights.shape := (n_time)"""
    
    time_dim_size = tensor_to_average.shape[time_dim]
    
    assert 0 <= time_dim < tensor_to_average.ndim, \
            'time_dim must be a number in [0, tensor_to_average.ndim)'
        
    # the tensor must be floating point in order for averaging to work properly
    tensor_to_average = tensor_to_average.float()
    
    if tensor_to_average.is_sparse:
        # we need to make a copy of the tensor (for now) since the value operations
        # happen in place, and we don't want to change the original tensor
        tensor_to_average = tensor_to_average.clone().coalesce()
        
    if weights is not None:
        weights = weights.float()
        assert weights.ndim == 1 and weights.shape[0] == tensor_to_average.shape[time_dim], \
                'weights.shape should equal the size of the time axis'
        assert torch.all(0 <= weights ), 'The weight vector should be positive'
        weights = weights / weights.sum()  # normalizing to probability vector and scaling by time_dim_size
        # multiplying weights along the time axis
        if tensor_to_average.is_sparse:
            value_multipliers_by_time = weights[tensor_to_average.indices()[time_dim]]
            tensor_to_average.values().mul_(value_multipliers_by_time)
            return torch.sparse.sum(tensor_to_average, dim=time_dim)
        else:
            # setting up weights so it can be broadcasted along the time_dim
            weight_reshape = ([1]*tensor_to_average.ndim)
            weight_reshape[time_dim] = -1
            tensor_to_average = tensor_to_average * weights.reshape(weight_reshape)
            return torch.sum(tensor_to_average, dim=time_dim)
            
    time_dim_size = tensor_to_average.shape[time_dim]
    if tensor_to_average.is_sparse:
        return torch.sparse.sum(tensor_to_average, dim=time_dim) / time_dim_size
    else:
        return tensor_to_average.sum(time_dim) / time_dim_size

def sparse_tensor_to_sparse_matrix(sparse_tensor, dim_to_keep=1):
    """This essentially acts as a general move + 2d reshaping of a sparse tensor to a sparse matrix where
    dim_to_keep is the dimension to hold constant and all others are flattened. 
    It is equivalent to the dense operation: `dense_tensor.moveaxis(dim_to_keep, 0).reshape(shape[dim_to_keep], -1).`
    The `sparse_tensor.ndim` must be greater or equal to 2, and the dim_to_keep must be \in [0, sparse_tensor.ndim]"""
    # In: sparse_tensor.shape = [dim_0, dim_1, ..., dim_k]
    # Out: sparse_matrix.shape = [dim_to_keep, product(dims)/dim_to_keep]
    assert 0 <= dim_to_keep < sparse_tensor.ndim, 'dim_to_keep must be a number in [0, sparse_tensor.ndim)'
    
    sparse_tensor = sparse_tensor.coalesce()
    
    # first we permute the sparse tensor indices so that the dim_to_keep becomes the first dim
    perm = list(range(sparse_tensor.ndim))
    perm.pop(dim_to_keep)
    perm.insert(0, dim_to_keep)
    
    indices = sparse_tensor.indices()
    perm_indices = indices[perm, :]
    
    perm_shape = torch.tensor(sparse_tensor.shape)[perm]
    return_matrix_shape = (perm_shape[0], torch.prod(perm_shape[1:]))
    
    # setting up strides
    dim_strides = torch.tensor([torch.prod(perm_shape[i:]).item() for i in range(2,sparse_tensor.ndim)] + [1])
    # raveling the indices from perm_dims[1:]
    matrix_col_indices = torch.sum(perm_indices[1:].T * dim_strides, axis=1)
    
    new_indices = torch.stack((perm_indices[0], matrix_col_indices))

    return torch.sparse_coo_tensor(new_indices, sparse_tensor.values(), size=return_matrix_shape).coalesce()

def sparse_1x1_convolution(sparse_tensor, weight_mat):
    # In: sparse_tensor.shape = [n_batch, n_channels, h, w]
    #     weight_mat.shape = [n_channels_in, n_channels_out]
    
    assert weight_mat.ndim == 2, 'weight_mat must has shape [n_channels, n_batch*h*w]'
    assert sparse_tensor.shape[1] == weight_mat.shape[0]
    # flattening sparse_tensor to sparse_matrix
    sparse_matrix = sparse_tensor_to_sparse_matrix(sparse_tensor, dim_to_keep=1)
    # convolving sparse_matrix
    convolved_matrix = sparse_matrix.t().mm(weight_mat).t()
    # reshaping convolved_matrix back to the original shape (with n_channels_out)
    prepermute_output_shape = (weight_mat.shape[1], sparse_tensor.shape[0],
                               sparse_tensor.shape[2], sparse_tensor.shape[3])
    convolved_matrix = convolved_matrix.reshape(prepermute_output_shape).moveaxis(1,0)
    
    return convolved_matrix


def create_random_sparse_tensor(size, n_values, arange=False):
    """A helper function for quickly creating sparse tensor of a specified size"""
    indices = torch.vstack([ torch.randint(0, s, size=(n_values, )) for s in size])
    if arange:
        values = torch.arange(n_values)
    else:
        values = torch.randn(n_values)
    
    return torch.sparse_coo_tensor(indices, values, size)

def get_sparse_slice(sparse_tensor, start, stop):
    """A simple helper slice function since sparse tensors don't allow slicing
    equivalent to the dense operation: tensor[start:stop]"""
    sparse_list = []
    for idx in range(start, stop):
        sparse_list.append(sparse_tensor[idx])
    return torch.stack(sparse_list)