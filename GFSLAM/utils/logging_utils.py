import rich
import torch

_log_styles = {
    "MonoGS": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "GSFlow": "bold blue",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="GSFlow"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)

def print_gpu_mem(tag=""):
    alloc = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    print(f"[{tag}] Allocated: {alloc:.2f} MB, Reserved: {reserved:.2f} MB")