#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Dict


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


def get_gpu_info() -> Dict[int, Dict[str, float]]:
    import nvidia_smi

    nvidia_smi.nvmlInit()
    gpu_num = nvidia_smi.nvmlDeviceGetCount()

    mems = {}
    for i in range(gpu_num):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free = info.total - info.used
        mems[i] = {"total": info.total, "used": info.used, "free": free}
    return mems


def get_gpu_mems() -> List[float]:
    assert check_import(
        "nvidia_smi"
    ), f"automatic gpu assignment requires nvidia_smi as dependency. Try `pip install trex[gpu]`"

    info = get_gpu_info()
    return [info[i]["free"] for i in range(len(info))]
