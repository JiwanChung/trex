#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from typing import Tuple

import click
import yaml

from trex.utils import get_topk, get_gpu_mems
from trex.run import run_cmd


def load_options():
    path = Path.home() / '.config/trex.yaml'
    if not path.is_file():
        src_path = Path(__file__).parent / 'trex.yaml'
        shutil.copyfile(src_path, str(path))

    with open(path) as f:
        options = yaml.safe_load(f)
    return options


@click.command()
@click.argument('gpus')
@click.option('-b', '--batch', default=False, is_flag=True)
@click.option('-a', '--auto', default=False, is_flag=True)
@click.option('-s', '--server', type=str, default=None, show_default=True)
@click.argument('command', nargs=-1, type=str)
def trex(gpus: str, batch: bool, auto: bool, server: str, command: Tuple[str]):
    if len(command) == 0:
        print("No command given")
        return

    options = load_options()
    is_slurm = shutil.which('srun') is not None
    is_slurm = is_slurm & options.get('flags', {}).get(
        'use_slurm_when_available', True)

    is_batch = len(command) == 1 and command[0].endswith('sh')
    is_batch = is_batch | batch

    if is_batch:
        if not Path(command[0]).is_file():
            print(f'No batch script file found: {command[0]}')
            exit()

    command = ' '.join(command)
    envs = {}

    if is_slurm:
        cmd = 'sbatch' if is_batch else 'srun'
        cmds = []
        if server is None:
            server == 'default'

        server_options = options.get('server', {})
        if server not in server_options:
            print(
                f'server is not specified in the configuration file: {server}')
            return
        opt = server_options[server]
        if 'p' in opt:
            cmds = [*cmds, *['-p', opt['p']]]
        if 'q' in opt:
            cmds = [*cmds, *['-q', opt['q']]]

        if gpus != 'x':
            cmds = [*cmds, f'--gres=gpu:{gpus}']
        else:
            envs['CUDA_VISIBLE_DEVICES'] = ''
        cmds = ' '.join(cmds)
        cmds = f'{cmds} {command}'
    else:
        shell = os.environ['SHELL']
        if gpus != 'x':
            flag = options.get('flags',
                               {}).get('local_automatic_gpu_assignment', False)
            flag = flag or auto
            flag = flag and ',' not in gpus
            if flag:
                try:
                    gpu_num = int(gpus)
                except Exception as e:
                    print(f'Invalid gpu setup: {gpus}')
                    return
                try:
                    gpu_mems = get_gpu_mems()
                except Exception as e:
                    print(e)
                    return
                gpus = get_topk(gpu_mems, gpu_num)
                gpus = ','.join([str(v) for v in gpus])
            envs['CUDA_VISIBLE_DEVICES'] = str(gpus)
        else:
            envs['CUDA_VISIBLE_DEVICES'] = ''

        if is_batch:
            cmds = f'{shell} {command}'
        else:
            cmds = command
    print(f'trex running command: ({cmds})')
    run_cmd(cmds, batch=is_batch, env=envs)
