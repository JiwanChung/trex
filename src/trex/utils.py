#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Dict, Any


def argsort(seq):
    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]


def get_topk(seq: Dict[Any, float], k: int = 1, largest: bool = True) -> List[float]:
    _seq = sorted(seq.items(), key=lambda x: x[1])

    if largest:
        _seq = reversed(_seq)
    _seq = list([x[0] for x in _seq])
    return _seq[:k]


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


def get_gpu_mems() -> Dict[int, float]:
    assert check_import(
        "nvidia_smi"
    ), "automatic gpu assignment requires nvidia_smi as dependency. Try `pip install trex[gpu]`"

    info = get_gpu_info()
    return {k: v["free"] for k, v in info.items()}
