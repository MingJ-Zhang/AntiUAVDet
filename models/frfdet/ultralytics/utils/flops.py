import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from ultralytics import YOLO
from thop import profile


def main(
        weight: str,
        resolution: int = 640,
        half: bool = False,
        mode: str = 'thop',
        name: str = ''
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.randn((1, 3, resolution, resolution)).to(device)
    if half:
        inputs = inputs.half()
    model = torch.load(weight, map_location=device)

    if isinstance(model, dict) and model.get('model'):
        model = model['model']
    elif model.get('ema'):
        model = model['ema']
    if name == 'YOLOv10':
        model.model[-1].cv2 = nn.ModuleList([nn.Identity() for i in model.model[-1].cv2])
        model.model[-1].cv3 = nn.ModuleList([nn.Identity() for i in model.model[-1].cv3])
    model = model.float()
    model.fuse()
    if half:
        model = model.half()

    model = model.eval().to(device)

    for name, param in model.named_parameters():
        param.requires_grad = True

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('Params', n_parameters / 1e6, 'M')

    # thop
    if mode == 'thop':
        flops, params = profile(model, inputs=(inputs,))
        print("FLOPs:", flops / 1e9, "GFLOPs")
    else:
        flops = FlopCountAnalysis(model, inputs)
        print("FLOPs:", flops.total() / 1e9, "GFLOPs")


if __name__ == '__main__':
    main(r'',
         resolution=640,
         half=True)