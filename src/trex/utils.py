#!/usr/bin/env python
# -*- coding: utf-8 -*-


def argsort(seq):
    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]


def get_topk(seq, k: int = 1, largest: bool = True):
    seq = argsort(seq)
    if largest:
        seq = reversed(seq)
    seq = list(seq)
    return seq[:k]


def check_import(name: str):
    import importlib.util
    spam_spec = importlib.util.find_spec(name)
    found = spam_spec is not None
    return found


def get_gpu_mems():
    assert check_import(
        'torch'
    ), f"automatic gpu assignment requires pytorch as dependency. Try `pip install trex[torch]`"

    import torch

    gpu_num = torch.cuda.device_count()
    free_mems = [torch.cuda.mem_get_info(device=i)[0] for i in range(gpu_num)]
    return free_mems
