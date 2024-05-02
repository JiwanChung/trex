#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess


def test_batch():
    path = str(Path(__file__).parent / 'ls.sh')
    out = subprocess.run(['trex', 'x', path], capture_output=True)
    print(out)


def test_run1():
    out = subprocess.run(['trex', 'x', 'ls'], capture_output=True)
    print(out)


def test_run2():
    path = str(Path(__file__).parent / 'ls.sh')
    out = subprocess.run(['trex', 'x', 'bash', path], capture_output=True)
    print(out)
