# [KDD 2024] Temporal Prototype-Aware Learning for Active Voltage Control on Power Distribution Networks

[![arXiv](https://img.shields.io/badge/arXiv-2406.17818-b31b1b.svg)](https://arxiv.org/abs/2406.17818)

Official codebase for the paper [Temporal Prototype-Aware Learning for Active Voltage Control on Power Distribution Networks](https://arxiv.org/abs/2406.17818).



## Overview

![model_fig](https://github.com/Canyizl/TPA-for-AVC/blob/main/paper_contents/model_new.png)

**Abstract:** Active Voltage Control (AVC) on the Power Distribution Networks (PDNs) aims to stabilize the voltage levels to ensure efficient and reliable operation of power systems. With the increasing integration of distributed energy resources, recent efforts have explored employing multi-agent reinforcement learning (MARL) techniques to realize effective AVC. Existing methods mainly focus on the acquisition of short-term AVC strategies, i.e., only learning AVC within the short-term training trajectories of a singular diurnal cycle. However, due to the dynamic nature of load demands and renewable energy, the operation states of real-world PDNs may exhibit significant distribution shifts across varying timescales (e.g., daily and seasonal changes). This can render those short-term strategies suboptimal or even obsolete when performing continuous AVC over extended periods. In this paper, we propose a novel temporal prototype-aware learning method, abbreviated as TPA, to learn time-adaptive AVC under short-term training trajectories. At the heart of TPA are two complementary components, namely multi-scale dynamic encoder and temporal prototype-aware policy, that can be readily incorporated into various MARL methods. The former component integrates a stacked transformer network to learn underlying temporal dependencies at different timescales of the PDNs, while the latter implements a learnable prototype matching mechanism to construct a dedicated AVC policy that can dynamically adapt to the evolving operation states. Experimental results on the AVC benchmark with different PDN sizes demonstrate that the proposed TPA surpasses the state-of-the-art counterparts not only in terms of control performance but also by offering model transferability.



## Installation

We install dependencies based on [MAPDN](https://github.com/Future-Power-Networks/MAPDN).

Please execute the following command:

```shell
conda env create -f environment.yml

conda activate mapdn
```



## Downloading the Dataset

The dataset is also provided by [MAPDN](https://github.com/Future-Power-Networks/MAPDN).

1. Download the data from the [link](https://drive.google.com/file/d/1-GGPBSolVjX1HseJVblNY3KoTqfblmLh/view?usp=sharing).
2. Unzip the zip file and you can see the following 3 folders:

    * `case33_3min_final`
    * `case141_3min_final`
    * `case322_3min_final`
3. Go to the directory `[Your own parent path]/TPA/environments/var_voltage_control/` and create a folder called `data`.
4. Move the 3 extracted folders by step 2 to the directory `[Your own parent path]/TPA/environments/var_voltage_control/data/`.



## Running experiments

#### Checkpoint

You can download our pre-trained checkpoints [here](https://drive.google.com/drive/folders/1W9EnhzmBDY8rt-3YZPjJPrL9WQlZMdcG?usp=sharing).

It should be noted that, based on the previous environments, we are confused that we can not reproduce the training results through the common seed-fixed method.

#### Training

You can train the model using the following command.

```bash
source activate mapdn

python train.py --alg tpamaddpg --alias tpa_maddpg_322 --mode distributed --scenario case322_3min_final --qweight 0.1 --voltage-barrier-type l1 --save-path trial/

python train.py --alg tpamaddpg --alias tpa_maddpg_141 --mode distributed --scenario case141_3min_final --qweight 0.01 --voltage-barrier-type l1 --save-path trial/

```

The meanings of the arguments:

* `--alg` is the MARL algorithm, e.g. `maddpg`, `matd3`, `tpamaddpg`, `tpamadt3`.
* `--alias` is the alias to distinguish different experiments.
* `--mode` is the mode of environment, e.g. `distributed`.
* `--scenario` is the power system on which you like to train, e.g. `case141_3min_final`, `case322_3min_final`.
* `--qweight` is the q_weight used in training. We recommend 0.01 for case141 and 0.1 for case322.
* `--voltage-barrier-type` is the voltage barrier function in training, e.g. `l1`, `l2`, `bowl`.
* `--save-path` is the path to save the model and configures.

#### Testing

After training, you can exclusively test your model to do the further analysis using the following command

```bash
python test.py --save-path trial/model_save --alg tpamaddpg --seed 53 --alias tpa_maddpg_322 --scenario case322_3min_final --qweight 0.1 --voltage-barrier-type l1 --test-mode test_data

python test.py --save-path trial/model_save --alg tpamaddpg --seed 53 --alias tpa_maddpg_322 --scenario case322_3min_final --qweight 0.1 --voltage-barrier-type l1 --test-mode year
```

If you use our pretrained ckpts, the command should be

```bash
python test.py --save-path trial/model_save --alg tpamaddpg --seed 53 --alias example322tpa --scenario case322_3min_final --qweight 0.1 --voltage-barrier-type l1 --test-mode test_data
```

The meanings of the arguments:

* `--alg` is the MARL algorithm, e.g. `maddpg`, `matd3`, `tpamaddpg`, `tpamadt3`.
* `--alias` is the alias to distinguish different experiments. We give the `example322tpa` as the pretrained ckpt for 322-bus.
* `--scenario` is the power system on which you like to train, e.g. `case141_3min_final`, `case322_3min_final`.
* `--qweight` is the q_weight used in training. We recommend 0.01 for case141 and 0.1 for case322.
* `--voltage-barrier-type` is the voltage barrier function in training, e.g. `l1`, `l2`, `bowl`.
* `--save-path` is the path to save the model and configures.
* `--test-mode` is the test mode, e.g. `test_data` is the previous testing mode on day cycle. `long`. `year` means longer testing cycles. 
* `--test-day` is the day that you would like to do the test. Note that it is only activated if the `--test-mode` is `single`.


## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{xu2024TPA,
  title     = {Temporal Prototype-Aware Learning for Active Voltage Control on Power Distribution Networks},
  author    = {Xu, Feiyang and Liu, Shunyu and Qing, Yunpeng and Zhou, Yihe and Wang, Yuwen and Song, Mingli},
  booktitle = {ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2024},
}
```

## Contact

Please feel free to contact me via email (<xufeiyang@zju.edu.cn>, <liushunyu@zju.edu.cn>) if you are interested in my research :)

