#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional

import click
import yaml

from trex.run import run_cmd
from trex.utils import get_gpu_mems, get_topk


def load_options():
    path = Path.home() / ".config/trex.yaml"
    if not path.is_file():
        src_path = Path(__file__).parent / "trex.yaml"
        shutil.copyfile(src_path, str(path))

    with open(path) as f:
        options = yaml.safe_load(f)
    return options


@click.command()
@click.argument("gpus")
@click.option(
    "-b",
    "--batch",
    default=False,
    is_flag=True,
    help="set to use sbatch for files not ending with .sh",
)
@click.option(
    "-i",
    "--index",
    default=False,
    is_flag=True,
    help="manual set GPU indices in local mode",
)
@click.option(
    "-o", "--output", default="none", help="slurm mode: sbatch log file output"
)
@click.option("-l", "--local-mode", is_flag=True, help="force local mode")
@click.option(
    "-s",
    "--server",
    type=str,
    default="default",
    show_default=True,
    help="server name specified in ~/.config/trex.yaml",
)
@click.option(
    "-a",
    "--allowed",
    type=str,
    default=None,
    show_default=True,
    help="Confine available GPUs to the given indices. e.g., 0,1,2,3",
)
@click.argument("command", nargs=-1, type=str)
def trex(
    gpus: str,
    batch: bool,
    index: bool,
    output: str,
    local_mode: bool,
    server: str,
    allowed: Optional[str],
    command: Tuple[str],
):
    if len(command) == 0:
        print("No command given")
        return
    options = load_options()

    allowed_gpus = (
        [int(gpu) for gpu in allowed.split(",")] if allowed is not None else None
    )
    if allowed_gpus is None:
        allowed_gpus = options.get("settings", {}).get("allowed_gpus", None)
    if allowed_gpus is not None:
        print(f"Confining available GPUs to: {allowed_gpus}")

    is_slurm = shutil.which("srun") is not None
    is_slurm = is_slurm & options.get("settings", {}).get(
        "use_slurm_when_available", True
    )
    if local_mode:
        is_slurm = False

    # parse args
    if is_slurm and index:
        print("manual gpu index is not supported in slurm mode")
        exit()

    # parse batch mode
    is_batch = len(command) == 1 and command[0].endswith("sh")
    is_batch = is_batch | batch
    if is_batch:
        if not Path(command[0]).is_file():
            print(f"No batch script file found: {command[0]}")
            exit()

    # parse gpu args
    if index and not is_slurm:
        # manually setting gpu indices
        use_gpus = all([gpu.isdigit() and int(gpu) != 0 for gpu in gpus.split(",")])
    else:
        use_gpus = gpus.isdigit() and int(gpus) != 0
    if not (use_gpus or gpus in ["0", "x"]):
        print(f"Invalid gpu setup: {gpus}")
        exit()

    command_str = " ".join(command)
    envs = {}
    if is_slurm:
        cmd = "sbatch" if is_batch else "srun"
        cmds = [cmd]
        if is_batch and output != "none":
            cmds = [*cmds, "-o", output]

        server_options = options.get("server", {})
        server_options = {str(k): v for k, v in server_options.items()}

        if not use_gpus and server == "default":
            server = "cpu_default"
            if server not in server_options:
                server = "default"

        if server not in server_options:
            print(server, server_options)
            print(f"server is not specified in the configuration file: {server}")
            return
        opt = server_options[server]
        if "p" in opt:
            cmds = [*cmds, *["-p", opt["p"]]]
        if "q" in opt:
            cmds = [*cmds, *["-q", opt["q"]]]

        if use_gpus:
            cmds = [*cmds, f"--gres=gpu:{gpus}"]

        cmds = " ".join(cmds)
        cmds = f"{cmds} {command_str}"
    else:
        shell = os.environ["SHELL"]
        if use_gpus:
            min_gpu_mem = options.get("settings", {}).get("min_gpu_mem", 1)
            min_gpu_mem = min_gpu_mem * 1024 * 1024  # convert to GB

            try:
                gpu_mems = get_gpu_mems()
            except Exception as e:
                print("failded to retrieve GPU informations")
                print(e)
                return
            gpu_mems = {k: v for k, v in gpu_mems.items() if v >= min_gpu_mem}
            if allowed_gpus is not None:
                gpu_mems = {k: v for k, v in gpu_mems.items() if k in allowed_gpus}

            if not index:
                gpu_num = int(gpus)

                if gpu_num > len(gpu_mems):
                    print(
                        f"requested {gpu_num} GPUs, but only {len(gpu_mems)} Free GPUs are available"
                    )
                    return
                gpus_li = get_topk(gpu_mems, gpu_num)
                gpus = ",".join([str(v) for v in sorted(gpus_li)])
            else:
                gpu_ids = [int(g) for g in gpus.split(",")]
                invalids = set(gpu_ids) - set(gpu_mems.keys())
                if len(invalids) > 0:
                    msg = f"requested GPU indices: {list(sorted(gpu_ids))}, but GPU {list(sorted(invalids))} are not available"
                    print(msg)
                    return

            envs["CUDA_VISIBLE_DEVICES"] = gpus
        else:
            envs["CUDA_VISIBLE_DEVICES"] = ""

        if is_batch:
            cmds = f"{shell} {command_str}"
        else:
            cmds = command_str
    print(f"trex: {{gpus: [{gpus}], command: ({cmds})}}")
    run_cmd(cmds, batch=is_batch and not is_slurm, env=envs)
