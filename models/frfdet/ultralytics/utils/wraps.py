"""
decorator
"""
import typing as t
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
from torchsummary import summary as ts
from torchinfo import summary as tis
from thop import profile
from .torch_utils import profile as yp

__all__ = ['log_profile' ]

def format_number(number):
    if number >= 1e9:  # Giga
        return f"{number / 1e9:.2f} G"
    elif number >= 1e6:  # Mega
        return f"{number / 1e6:.2f} M"
    elif number >= 1e3:  # Kilo
        return f"{number / 1e3:.2f} K"
    else:
        return str(number)

def print_parameters(model):
    print("Parameter Count by Layer:")
    table = parameter_count_table(model)  # Adjust `max_depth` as needed
    print(table)

def log_profile(
        mode: str,
        shape: t.Tuple = (3, 640, 640),
        batch_size: int = 1,
        show_detail: bool = False,
):
    """
    print some profile of model, such as Params, FLOPs
    """
    def wrapper(func: t.Callable):
        def inner(*args, **kwargs):
            model = func(*args, **kwargs)
            if getattr(model, 'model'):
                m = model.model
            device = kwargs.get('device', 'cpu')
            m = m.to(device)
            x = torch.randn((batch_size, *shape), device=device)

            if mode == 'thop':
                flops, params = profile(m, (x,))
                print(f"FLOPs={str(flops / 1e9)}G")
                print(f"params={str(params / 1e6)}M")
            elif mode == 'torchsummary':
                ts(m, shape, batch_size=batch_size, device=device)
            elif mode == 'thopinfo':
                tis(m, input_size=x.shape)
            elif mode == 'fvcore':
                flops = FlopCountAnalysis(model, inputs=x)
                params = parameter_count(model)
                formatted_flops = format_number(flops.total())
                formatted_params = format_number(params[""])
                print(f"FLOPs: {formatted_flops}")  # 打印总FLOPs
                print(f"Params: {formatted_params}")  # 打印总参数量

                # 打印每层的参数量
                print_parameters(model)

                # 如果你想查看每层的FLOPs
                print("FLOPs by Module:")
                for item in flops.by_module().items():
                    print(f"{item[0]}: {format_number(item[1])}")

            if show_detail:
                _x = torch.randn(batch_size, *shape, device=device)
                yp(_x, m, n=6)  # profile over 100 iterations
            return model
        return inner
    return wrapper
