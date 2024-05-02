# Tremendous REmote eXecutor

[![PyPI - Version](https://img.shields.io/pypi/v/trex.svg)](https://pypi.org/project/trex)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/trex.svg)](https://pypi.org/project/trex)

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

## Installation

```bash
pip install trex
```

## Usage

Simply run the command.

```bash
# running job in slurm

trex x echo "slurm job"

# running job in local machine

trex x echo "local job"
```

Where `x` means no GPU utilized.

Yes, they are the same. By default, `trex` runs on local machine when slurm is not installed.
You can adjust this behaviour by modifying the configuration file.

To use GPUs,

```bash
# running job in slurm

trex 2 echo "slurm GPUs"

# running job in local machine

trex 0,1 echo "local GPUs"
```

When running in a local machine, you would designate the GPU indices. In slurm environment, just input the required number of GPUs.
This behaviour can also be changed to align local to slurm remote setting by automatically assigning empty GPUs.

In slurm environment, you may also specify your remote server setups.

Modify `$HOME/.config/trex.yaml` by adding your server specifications.

```yaml
servers:
    my_server1:
        p: a6000
        q: big_qos
```

Then, run the same command as above with `-s/--server` flag set.

```bash
trex 2 -s my_server1 echo "roughly equal to srun -p a6000 -q big_qos --gres=gpu:2 echo"
```

Finally, `trex` automatically routes jobs to either `sbatch` or `srun`.
The criterion is simple: you pass a shell file, you get `sbatch` (or `sh $FILE` for local machines).

```bash
trex my_job.sh  # sbatch-like
trex bash my_job.sh  # srun-like
```

Also, you can explictly use `sbatch`-like mode by specifing the `-b/--batch` flag.

```bash
trex x -b my_job  # sbatch-like command for file my_job
```

By the way, I suggest aliasing `trex` to `x` for an even easier access.

## Configuration

Modify `$HOME/.config/trex.yaml`.

## License

`trex` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
