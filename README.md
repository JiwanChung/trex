# 🦖 Tremendous REmote eXecutor

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
pip install git+https://github.com/JiwanChung/trex
```

## Usage

1. Basic usage

Simply run the command.

```bash
# running job in slurm

trex x echo "slurm job"

# running job in local machine

trex x echo "local job"
```

Where `x` means no GPU utilized.

Yes, they are the same. By default, `trex` runs on local machine when slurm is not installed.
You can adjust this behaviour by modifying the `use_slurm_when_available` flag in the configuration file.
Also, you can set `-l` flag when running the command to override the default behaviour and force local mode.

2. Launching GPU Jobs

```bash
# running job w/ 2 gpus in slurm

trex 2 echo "slurm GPUs"

# running job w/ 2 gpus in local machine

trex 2 echo "local GPUs"

# in local setup, you can use -i/--index flag to manually designate the gpu indices

trex -i 0,2 echo "GPU 0 and 2"
```

You should specify your remote server setups when using `trex` in slurm environments.
Modify `$HOME/.config/trex.yaml` by adding your server specifications.

```yaml
servers:
    default:
        p: x
        q: x
    cpu_default:
        p: x
        q: x
    my_server1:
        p: a6000
        q: big_qos
```

The `default` and `cpu_default` servers are special entries that are used by default when no server is specified by the `-s/--server` argument.

```bash
trex 2 -s my_server1 echo "roughly equal to srun -p a6000 -q big_qos --gres=gpu:2 echo"
```

3. Batch-mode

`trex` automatically routes jobs to either `sbatch` or `srun`.
The criterion is simple: you pass a shell file ending with `.sh`, you get `sbatch` (or `sh $FILE` for local machines).

```bash
trex my_job.sh  # sbatch-like
trex bash my_job.sh  # srun-like
```

Also, you can explictly use `sbatch`-like mode by specifing the `-b/--batch` flag.

```bash
trex x -b my_job  # sbatch-like command for file my_job
```

As a side note, I suggest aliasing `trex` to `x` for an even easier access :).

## Configuration

After running the command first time, you will see the example configuration at `$HOME/.config/trex.yaml`.
Modify the values accordingly.

Here are some suggestions:

1. Servers (slurm mode)

```yaml
    server:
        default:
            p: x
            q: x
```

Aside from the reserved keys (`default`, `cpu_default`), the server keys can be named arbitrarily to your liking.
Just make sure that the `p` and `q` values are set up correctly (they should be equivalent to what you would send to `srun` command with the `-p` and `-q` arguments.).

2. Turning off slurm by default

Even when you are on an environment with `slurm` installed, you can turn off slurm mode by default.

```yaml
    settings:
        use_slurm_when_available: False
```

3. Confining GPU indices (local mode)

Sometimes you share the same remote machine with co-workers and make an arrangment to divide GPU resources based on their indices.
I've got you covered in those cases as well. For example, when you are assigned GPUs `[0,4,8]` modify the configuation file as follows:

```yaml
    settings:
        allowed_gpus: [0,4,8]
```

Then everything would work as expected. `trex` will not use the other GPUs in automatic assignment and raise errors when forced to use them via explicit `-i/--index` arguments.

This option is also available on-the-fly as the `-a/--allowed` argument. In that case, provide indices with `,` as the separator.

4. Muting status message.

```yaml
    settings:
        mute_status: True
```

Now you won't see the status report every time you run `trex`.

## License

`trex` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.
