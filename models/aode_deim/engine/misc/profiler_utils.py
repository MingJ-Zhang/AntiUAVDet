"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    # Non-square detectors cache positional embeddings for eval_spatial_size.
    # Profiling them with a synthetic square image makes the flattened feature
    # length disagree with that cache before training even starts.
    eval_spatial_size = cfg.yaml_cfg.get('eval_spatial_size')
    if eval_spatial_size is not None:
        height, width = eval_spatial_size
        input_shape = (1, 3, height, width)
    else:
        base_size = cfg.train_dataloader.collate_fn.base_size
        input_shape = (1, 3, base_size, base_size)

    # AoDE keeps the two-input LQE call in the decoder. The upstream deploy()
    # path replaces LQE with nn.Identity (one input), so it is not suitable for
    # profiling this training model. Evaluation mode is sufficient for FLOPs.
    model_for_info = copy.deepcopy(cfg.model).eval()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
