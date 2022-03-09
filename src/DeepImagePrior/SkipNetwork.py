# Based on: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/models/skip.py

import torch
import torch.nn as nn
from .common import *

def SkipNetwork(in_channels = 3,
                out_channels = 3,
                hidden_dims_down = [16, 32, 64, 128, 128],
                hidden_dims_up = [16, 32, 64, 128, 128],
                hidden_dims_skip = [4, 4, 4, 4, 4],
                filter_size_down = 3,
                filter_size_up = 3,
                filter_skip_size = 1,
                sigmoid_output = True,
                tanh_output = False,
                bias = True,
                padding_mode='zero',
                upsample_mode='nearest',
                downsample_mode='stride',
                act_fun='LeakyReLU',
                need1x1_up=True
               ):
    """Assembles encoder-decoder with skip connections.
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        padding_mode (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
    """
    assert len(hidden_dims_down) == len(hidden_dims_up) == len(hidden_dims_skip)

    n_scales = len(hidden_dims_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = in_channels
    for i in range(len(hidden_dims_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if hidden_dims_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(hidden_dims_skip[i] + (hidden_dims_up[i + 1] if i < last_scale else hidden_dims_down[i])))

        if hidden_dims_skip[i] != 0:
            skip.add(conv(input_depth, hidden_dims_skip[i], filter_skip_size, bias=bias, padding_mode=padding_mode))
            skip.add(bn(hidden_dims_skip[i]))
            skip.add(act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, hidden_dims_down[i], filter_size_down[i], 2, bias=bias, padding_mode=padding_mode, downsample_mode=downsample_mode[i]))
        deeper.add(bn(hidden_dims_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(hidden_dims_down[i], hidden_dims_down[i], filter_size_down[i], bias=bias, padding_mode=padding_mode))
        deeper.add(bn(hidden_dims_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(hidden_dims_down) - 1:
            # The deepest
            k = hidden_dims_down[i]
        else:
            deeper.add(deeper_main)
            k = hidden_dims_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(hidden_dims_skip[i] + k, hidden_dims_up[i], filter_size_up[i], 1, bias=bias, padding_mode=padding_mode))
        model_tmp.add(bn(hidden_dims_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            model_tmp.add(conv(hidden_dims_up[i], hidden_dims_up[i], 1, bias=bias, padding_mode=padding_mode))
            model_tmp.add(bn(hidden_dims_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = hidden_dims_down[i]
        model_tmp = deeper_main

    model.add(conv(hidden_dims_up[0], out_channels, 1, bias=bias, padding_mode=padding_mode))
    
    if sigmoid_output:
        model.add(nn.Sigmoid())
    elif tanh_output:
        model.add(nn.Tanh())

    return model


def SingleNetwork(in_channels = 3,
                  out_channels = 3,
                  hidden_dims = [16, 32, 64, 128, 128],
                  filter_size = 3,
                  input_filter_size = None,
                  sigmoid_output = True,
                  bias = True,
                  padding_mode='zero',
                  act_fun='LeakyReLU'
               ):
    """Assembles encoder-decoder with skip connections.
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        padding_mode (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
    """
    
    n_steps = len(hidden_dims)    
        
    if not (isinstance(filter_size, list) or isinstance(filter_size, tuple)) :
        filter_size = [filter_size]*(n_steps)

    model = nn.Sequential()
    input_depth = in_channels
    
    for i in range(n_steps):
        
        deeper = nn.Sequential(conv(input_depth,
                                    hidden_dims[i],
                                    filter_size[i],
                                    bias=bias, padding_mode=padding_mode),
                               #bn(hidden_dims[i]),
                               act(act_fun),
                               conv(hidden_dims[i],
                                    hidden_dims[i],
                                    filter_size[i],
                                    bias=bias, padding_mode=padding_mode),
                               #bn(hidden_dims[i]),
                               act(act_fun)
                              )
        
        model.add(deeper)
        input_depth = hidden_dims[i]            

    model.add(conv(input_depth, out_channels, 1, bias=bias, padding_mode=padding_mode))
    
    if sigmoid_output:
        model.add(nn.Sigmoid())

    return model