#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from typing import Tuple

import click
import yaml

from trex.run import run_cmd
from trex.utils import get_gpu_mems, get_topk


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
@click.option('-b',
              '--batch',
              default=False,
              is_flag=True,
              help='set to use sbatch for files not ending with .sh')
@click.option('-a',
              '--auto',
              default=False,
              is_flag=True,
              help='local mode: auto-assign gpus')
@click.option('-o',
              '--output',
              default='none',
              help='slurm mode: sbatch log file output')
@click.option('-l', '--local-mode', is_flag=True, help='force local mode')
@click.option('-s',
              '--server',
              type=str,
              default='default',
              show_default=True,
              help='server name specified in ~/.config/trex.yaml')
@click.argument('command', nargs=-1, type=str)
def trex(gpus: str, batch: bool, auto: bool, output: str, local_mode: bool,
         server: str, command: Tuple[str]):
    if len(command) == 0:
        print("No command given")
        return

    options = load_options()
    is_slurm = shutil.which('srun') is not None
    is_slurm = is_slurm & options.get('flags', {}).get(
        'use_slurm_when_available', True)
    if local_mode:
        is_slurm = False

    is_batch = len(command) == 1 and command[0].endswith('sh')
    is_batch = is_batch | batch

    if is_batch:
        if not Path(command[0]).is_file():
            print(f'No batch script file found: {command[0]}')
            exit()

    use_gpus = gpus.isdigit() and int(gpus) != 0

    command = ' '.join(command)
    envs = {}
    if is_slurm:
        cmd = 'sbatch' if is_batch else 'srun'
        cmds = [cmd]
        if is_batch and output != 'none':
            cmds = [*cmds, '-o', output]

        server_options = options.get('server', {})
        server_options = {str(k): v for k, v in server_options.items()}

        if not use_gpus and server == 'default':
            server = 'cpu_default'
            if server not in server_options:
                server = 'default'

        if server not in server_options:
            print(server, server_options)
            print(
                f'server is not specified in the configuration file: {server}')
            return
        opt = server_options[server]
        if 'p' in opt:
            cmds = [*cmds, *['-p', opt['p']]]
        if 'q' in opt:
            cmds = [*cmds, *['-q', opt['q']]]

        if use_gpus:
            cmds = [*cmds, f'--gres=gpu:{gpus}']
        else:
            envs['CUDA_VISIBLE_DEVICES'] = ''
        cmds = ' '.join(cmds)
        cmds = f'{cmds} {command}'
    else:
        shell = os.environ['SHELL']
        if use_gpus:
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
    run_cmd(cmds, batch=is_batch and not is_slurm, env=envs)
