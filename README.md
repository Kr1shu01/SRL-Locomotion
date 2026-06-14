# SRL-Locomotion

Code release accompanying the paper:

**SRL: Combining SLIP Model and Reinforcement Learning for Agile Robotic Jumping**

---

## Overview

This repository provides the core implementation of the proposed SRL (SLIP-guided Reinforcement Learning) framework for agile legged robot jumping.

SRL combines the physical priors of the Spring-Loaded Inverted Pendulum (SLIP) model with reinforcement learning to improve learning efficiency, robustness, and adaptability in dynamic jumping tasks.

The released code includes the key algorithmic components used in our work, including:

* SLIP-guided locomotion controllers
* Unity simulation assets and robot models
* Isaac Gym task environments
* MuJoCo sim-to-sim deployment scripts
* Demonstration videos

This repository is intended to illustrate the core ideas and implementation details of the SRL framework.

---

## Repository Structure

```text
SRL-Locomotion
├── unity/
│   ├── prefabs/
│   ├── urdf/
│   ├── Go2Agent.cs
│   ├── Go2Step.cs
│   ├── X02Agent.cs
│   └── configuration.yaml
│
├── isaacgym/
│   ├── envs/
│   └── scripts/
│
├── mujoco/
│   └── sim2sim_mujoco.py
│
└── videos/
```

---

## Included Components

### Unity

Contains robot assets, prefabs, configuration files, and controller implementations used for locomotion and jumping experiments.

Supported robots:

* Unitree Go2
* X02-lite humanoid robot

### Isaac Gym

Contains task environments and training-related scripts implementing the SRL framework.

The released code focuses on the key modifications beyond standard Isaac Gym training pipelines.

### MuJoCo

Provides sim-to-sim deployment scripts used for policy validation and transfer experiments.

---

## Demonstration Videos

The repository includes representative experimental demonstrations:

* Fixed-distance jumping
* Random-distance jumping
* Stepping and locomotion behaviors
* Biped and quadruped platforms

---

## Notes

This repository releases the core implementation and experimental assets associated with the paper.

Some platform-specific components, hardware interfaces, training infrastructure, and third-party dependencies are not included.

Researchers interested in the SRL framework can refer to the released source code and the accompanying paper for implementation details.

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{hu2026srl,
  title={SRL: Combining SLIP Model and Reinforcement Learning for Agile Robotic Jumping},
  author={Hu, Xiaowen and Ye, Linqi and others},
  journal={Robotics and Autonomous Systems},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgements

This work was conducted at Shanghai University and focuses on bio-inspired locomotion, reinforcement learning, and agile motion control for legged robots.
