#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess


def run_cmd(cmd, batch: bool = True, env: dict = {}):
    prev_env = os.environ.copy()
    env = {**prev_env, **env}
    subprocess.run(cmd, shell=True, env=env)
    # subprocess.run(cmd, shell=not batch, env=env)
