# Yet Another Pytorch Distributed MobileNetV2-based Networks Implementation

This repo reproduces some [MobileNetV2](https://arxiv.org/abs/1801.04381)-based networks which reach similar or even higher performance than the original reported results.


## Model Zoo

All of the following models are trained by [MnasNet](https://arxiv.org/abs/1807.11626) training schedule.

|                    | FLOPs | Parameters | Top-1 Report | Top-1 | Top-1 Calib |
| ---                | ---   | ---        | ---          | ---   | ---         |
| MobileNetV2        | 300M  | 3.4M       | 72.0         | 73.6  | -           |
| Proxyless (mobile) | 320M  | 4.1M       | 74.6         | 74.9  | 75.1        |
| SinglePath         | 334M  | 4.4M       | 75.0         | 75.0  | 75.1        |
| AtomNAS-A          | 258M  | 3.9M       | -            | -     | 74.6        |
| AtomNAS-B          | 326M  | 4.4M       | -            | -     | 75.5        |
| AtomNAS-C          | 360M  | 4.7M       | -            | -     | 75.9        |
| AtomNAS-A+         | 260M  | 4.7M       | -            | -     | 76.3        |
| AtomNAS-B+         | 329M  | 5.5M       | -            | -     | 77.2        |
| AtomNAS-C+         | 363M  | 5.9M       | -            | -     | 77.6        |

AtomNAS series comes from our ICLR 2020 paper [AtomNAS: Fine-Grained End-to-End Neural Architecture Search](https://openreview.net/forum?id=BylQSxHFwr) whose code resides in [AtomNAS](https://github.com/meijieru/AtomNAS).

Pretrained Models could be downloaded from [onedrive](https://1drv.ms/u/s!Alk-ml3frR0Iy0ItEpx6KluA6HOD?e=angPfD)

## Setup

### Distributed Training

Set the following ENV variable:
```
$DATA_ROOT: Path to data root
$METIS_WORKER_0_HOST: IP address of worker 0
$METIS_WORKER_0_PORT: Port used for initializing distributed environment
$METIS_TASK_INDEX: Index of task
$ARNOLD_WORKER_NUM: Number of workers
$ARNOLD_WORKER_GPU: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES `)
$ARNOLD_OUTPUT: Output directory
```

### Non-Distributed Training (Not Recommend)

Set the following ENV variable:
```
$DATA_ROOT: Path to data root
$ARNOLD_WORKER_GPU: Number of GPUs (NOTE: should exactly match local GPU numbers with `CUDA_VISIBLE_DEVICES `)
$ARNOLD_OUTPUT: Output directory
```


## Training

Take mobilenet v2 as example, for distributed training:
```bash
bash scripts/run.sh ./apps/mobilenet/mobilenet_v2_mnas.yml
```
For non-distributed training:
```bash
bash scripts/run_non_distributed_no_copy.sh ./apps/mobilenet/mobilenet_v2_mnas.yml
```


## Testing

For MobileNet, Proxyless, SinglePathNAS:
```bash
TRAIN_CONFIG=$(realpath {{train_config_path}}) ATOMNAS_VAL=True bash scripts/run.sh apps/eval/eval.yml --pretrained {{ckpt_path}}
```

For AtomNAS:
```bash
FILE=$(realpath {{log_dir_path}}) checkpoint=ckpt ATOMNAS_VAL=True bash scripts/run.sh apps/eval/eval_shrink.yml
```

For AtomNAS+:
```bash
TRAIN_CONFIG=$(realpath {{train_config_path}}) ATOMNAS_VAL=True bash scripts/run.sh apps/eval/eval_se.yml --pretrained {{ckpt_path}}
```

## Related Info

1. Requirements
    - See `requirements.txt`

1. Environment
    - The code is developed using python 3. NVIDIA GPUs are needed. The code is developed and tested using 4 servers with 32 NVIDIA V100 GPU cards. Other platforms or GPU cards are not fully tested.

1. Dataset
    - Prepare ImageNet data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
    - Optional: Generate lmdb dataset by `utils/lmdb_dataset.py`. If not, please overwrite `dataset:imagenet1k_lmdb` in yaml to `dataset:imagenet1k`.
    - The directory structure of `$DATA_ROOT` should look like this:
        ```
        ${DATA_ROOT}
        ├── imagenet
        └── imagenet_lmdb
        ```

1. Miscellaneous
    - The codebase is a general ImageNet training framework using yaml config with several extension under `apps` dir, based on PyTorch.
        - YAML config with additional features
            - `${ENV}` in yaml config.
            - `_include` for hierachy config.
            - `_default` key for overwriting.
            - `xxx.yyy.zzz` for partial overwriting.
        - `--{{opt}} {{new_val}}` for command line overwriting.


## Acknowledgment

This repo is based on [slimmable_networks](https://github.com/JiahuiYu/slimmable_networks) and benefits from the following projects
- [apex](https://github.com/NVIDIA/apex)
- [Efficient-PyTorch](https://github.com/Lyken17/Efficient-PyTorch)

Thanks the contributors of these repos!

## Citation

If you find this work or code is helpful in your research, please cite:
```
@inproceedings{
    mei2020atomnas,
    title={Atom{\{}NAS{\}}: Fine-Grained End-to-End Neural Architecture Search},
    author={Jieru Mei and Yingwei Li and Xiaochen Lian and Xiaojie Jin and Linjie Yang and Alan Yuille and Jianchao Yang},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=BylQSxHFwr}
}
```
