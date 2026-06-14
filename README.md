# SRL-Locomotion

Code release accompanying the paper:

**SRL: Combining SLIP Model and Reinforcement Learning for Agile Robotic Jumping**

<p align="center">
  <img src="media/frame.png" width="1000">
</p>

<p align="center">
  <em>
  Figure 1. Overview of the proposed SRL framework. The framework integrates a SLIP-based motion planner, a reinforcement learning controller, and an adaptive fusion module to achieve agile and robust jumping behaviors on both bipedal and quadrupedal robots.
  </em>
</p>

---

## Overview

This repository provides the core implementation of the proposed **SLIP-guided Reinforcement Learning (SRL)** framework for agile legged robot jumping.

SRL combines the physical priors of the Spring-Loaded Inverted Pendulum (SLIP) model with reinforcement learning to improve learning efficiency, robustness, and adaptability in dynamic jumping tasks.

The framework integrates:

- SLIP-based feedforward motion generation
- Reinforcement learning-based feedback control
- Adaptive feedforward-feedback action fusion
- Curriculum learning strategies
- Staged reward shaping
- Sim-to-sim transfer and deployment

The released code includes the key algorithmic components and experimental assets used in our work, providing a reference implementation of the SRL framework.

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
└── media/
    ├── frame.png
    ├── Go2_F.mp4
    ├── Go2_R.mp4
    ├── Go2_Step.mp4
    ├── SRL.mp4
    ├── X02_F.mp4
    └── X02_R.mp4
```

---

## Included Components

### Unity

Contains robot assets, URDF models, prefabs, configuration files, and controller implementations used for locomotion and jumping experiments.

**Supported robot platforms**

- Unitree Go2 quadruped robot
- X02-lite biped robot

**Key files**

- `Go2Agent.cs` – Go2 jumping controller
- `Go2Step.cs` – Go2 stepping controller
- `X02Agent.cs` – X02 jumping controller
- `configuration.yaml` – controller configuration

### Isaac Gym

Contains task environments and training-related scripts implementing the SRL framework.

The released code focuses on the key modifications beyond standard Isaac Gym training pipelines, including:

- SRL-specific environment design
- Reward shaping mechanisms
- SLIP-guided policy training
- Task-specific controller integration

These components contain the main algorithmic contributions of the proposed method.

### MuJoCo

Provides sim-to-sim deployment scripts used for policy validation and transfer experiments.

**Included file**

- `sim2sim_mujoco.py`

This module demonstrates the deployment pipeline used for sim-to-sim transfer experiments reported in the paper.

---

## Demonstration Videos

The `media/` directory contains representative demonstrations of the proposed SRL framework.

### Quadruped (Go2)

- `Go2_F.mp4` — Fixed-distance jumping
- `Go2_R.mp4` — Random-distance jumping
- `Go2_Step.mp4` — Stepping and locomotion behavior

### Biped (X02-lite)

- `X02_F.mp4` — Fixed-distance jumping
- `X02_R.mp4` — Random-distance jumping

### Framework Demonstration

- `SRL.mp4` — Overview of the SRL framework and experimental results

---

## Main Contributions

The proposed SRL framework combines:

- Physics-inspired SLIP motion planning
- Reinforcement learning-based feedback control
- Feedforward-feedback action fusion
- Curriculum learning strategies
- Staged reward shaping
- Sim-to-sim transfer capability

Compared with purely model-based or purely learning-based approaches, SRL leverages both physical priors and adaptive policy learning to achieve robust and agile jumping behaviors.

The framework has been validated on both bipedal and quadrupedal robot platforms across multiple jumping tasks.

---

## Notes

This repository releases the core implementation and experimental assets associated with the paper.

Some platform-specific components, hardware interfaces, training infrastructure, pretrained models, and third-party dependencies are not included in this release.

Therefore, this repository should be regarded as a **reference implementation** of the SRL framework rather than a complete reproduction package.

Researchers interested in reproducing the method are encouraged to refer to both the released source code and the accompanying paper for additional implementation details.

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

See the `LICENSE` file for details.

---

## Acknowledgements

This work was conducted at Shanghai University and focuses on bio-inspired locomotion, reinforcement learning, and agile motion control for legged robots.

We thank the open-source robotics community for making legged locomotion research more accessible and reproducible.
