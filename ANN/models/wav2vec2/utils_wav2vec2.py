import torch

def _downsample_output(x : torch.Tensor,nb_downsample : int):
    x = x.transpose(-2, -1)  # repermute to (layer, neuron, time) or (layer,batch,neuron,time)
    ## efficient downsampling mean computing:
    # unravel the array to the target sampling rate, removing the last elements.
    # use the mean over all remaining elements for the last:
    size_window = x.shape[-1] // nb_downsample
    to_remove = x.shape[-1] % nb_downsample
    y = x[..., :(x.shape[-1] - to_remove)].reshape(x.shape[:-1] + (nb_downsample, size_window))
    y_downsampled = torch.mean(y, dim=-1)
    last_window = torch.sum(x[..., x.shape[-1] - to_remove:], dim=-1)
    y_downsampled[..., -1] = (y_downsampled[..., -1] * size_window + last_window) / (to_remove + size_window)

    return y_downsampled

