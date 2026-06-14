# SRL-Locomotion

Official implementation of the **SLIP-guided Reinforcement Learning (SRL) Framework** for agile legged robot jumping.

---

## Overview

This repository contains the implementation of the proposed SRL framework, which integrates the Spring-Loaded Inverted Pendulum (SLIP) model with reinforcement learning to achieve agile and robust jumping behaviors for legged robots.

The framework combines model-based feedforward planning and learning-based feedback control, enabling efficient policy learning and strong adaptability across different jumping tasks.

The proposed method has been validated on both bipedal and quadrupedal robot platforms in simulation and demonstrates robust performance over a wide range of jumping distances.

---

## Key Features

* SLIP-based feedforward motion generation
* Reinforcement learning-based feedback control
* Adaptive fusion of feedforward and feedback actions
* Curriculum learning strategy
* Staged reward shaping
* Support for both bipedal and quadrupedal robots
* Webots simulation environment
* Sim-to-real deployment capability

---

## Framework Architecture

The SRL framework consists of three major components:

1. **SLIP Motion Planner**

   * Generates physically interpretable feedforward commands based on the SLIP model.

2. **Reinforcement Learning Controller**

   * Learns adaptive feedback policies from robot state observations.

3. **Fusion Module**

   * Integrates feedforward and feedback actions to produce final control commands.

A schematic illustration of the framework is provided in the paper.

---

## Repository Structure

```text
SRL-Locomotion
├── controllers/          # Robot controllers
├── environments/         # Simulation environments
├── reward_functions/     # Reward definitions
├── training/             # Training configurations
├── models/               # Network architectures
├── scripts/              # Training and evaluation scripts
├── webots/               # Webots worlds and assets
├── docs/                 # Documentation and figures
├── checkpoints/          # Pretrained models (optional)
└── README.md
```

---

## Requirements

* Python 3.10+
* PyTorch
* NumPy
* Stable-Baselines3
* Webots

Additional dependencies can be installed via:

```bash
pip install -r requirements.txt
```

---

## Training

### Biped Robot

```bash
python scripts/train.py --robot biped
```

### Quadruped Robot

```bash
python scripts/train.py --robot quadruped
```

---

## Evaluation

Evaluate a trained policy using:

```bash
python scripts/evaluate.py --checkpoint path/to/model.pt
```

---

## Reproducibility

To facilitate reproducibility, this repository provides:

* Source code of the SRL framework
* Training scripts
* Hyperparameter configurations
* Reward function definitions
* Simulation environments
* Evaluation scripts

Detailed parameter settings used in the paper are provided in the corresponding configuration files.

---

## Results

Representative experimental results are reported in the accompanying paper.

The proposed SRL framework demonstrates:

* Improved jumping stability
* Enhanced learning efficiency
* Strong generalization across jumping distances
* Successful transfer to different robot morphologies

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{SRL2026,
  title={SLIP-guided Reinforcement Learning for Agile Legged Robot Jumping},
  author={Author Names},
  journal={Journal Name},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

For questions, suggestions, or collaborations, please open an issue or contact the corresponding authors.

---

## Acknowledgements

This work was conducted at Shanghai University and was supported by research efforts in legged robot locomotion, reinforcement learning, and bio-inspired motion control.
