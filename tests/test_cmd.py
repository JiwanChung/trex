#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import subprocess


def test_batch():
    path = str(Path(__file__).parent / "ls.sh")
    trex_out = subprocess.run(["trex", "x", path], capture_output=True).stdout
    base_out = subprocess.run(["ls"], capture_output=True).stdout
    assert base_out in trex_out


def test_run1():
    trex_out = subprocess.run(["trex", "x", "ls"], capture_output=True).stdout
    base_out = subprocess.run(["ls"], capture_output=True).stdout
    assert base_out in trex_out


def test_args():
    trex_out = subprocess.run(["trex", "x", "ls", "-a"], capture_output=True).stdout
    base_out = subprocess.run(["ls", "-a"], capture_output=True).stdout
    print("---")
    print(trex_out)
    print(base_out)
    print("---")
    assert base_out in trex_out


def test_run2():
    path = str(Path(__file__).parent / "ls.sh")
    trex_out = subprocess.run(["trex", "x", "bash", path], capture_output=True).stdout
    base_out = subprocess.run(["ls"], capture_output=True).stdout
    assert base_out in trex_out


def test_gpu_batch():
    path = str(Path(__file__).parent / "check_gpu.sh")
    prev_env = os.environ.copy()
    env = {**prev_env, "CUDA_VISIBLE_DEVICES": "1,2"}
    trex_out = subprocess.run(
        ["trex", "-i" "1,2", "bash", path], capture_output=True
    ).stdout
    base_out = subprocess.run(
        ["python", "check_gpu.py"], capture_output=True, env=env
    ).stdout
    assert base_out in trex_out


def test_gpu_run():
    path = str(Path(__file__).parent / "check_gpu.sh")
    prev_env = os.environ.copy()
    env = {**prev_env, "CUDA_VISIBLE_DEVICES": "1,2"}
    trex_out = subprocess.run(
        ["trex", "-i", "1,2", "bash", path], capture_output=True
    ).stdout
    base_out = subprocess.run(
        ["python", "check_gpu.py"], capture_output=True, env=env
    ).stdout
    assert base_out in trex_out


def test_gpu_run_auto():
    path = str(Path(__file__).parent / "check_gpu.sh")
    prev_env = os.environ.copy()
    env = {**prev_env, "CUDA_VISIBLE_DEVICES": "1,2"}
    trex_out = subprocess.run(["trex", "2", "bash", path], capture_output=True).stdout
    base_out = subprocess.run(
        ["python", "check_gpu.py"], capture_output=True, env=env
    ).stdout
    assert base_out in trex_out
