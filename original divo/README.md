# DIVO: Diverse Policy Learning via Random Obstacle Deployment

**"Diverse Policy Learning via Random Obstacle Deployment for Zero-Shot Adaptation"**  
Seokjin Choi, Yonghyeon Lee, Seungyeon Kim, Che-Sang Park, Himchan Hwang, and Frank C. Park  
*IEEE Robotics and Automation Letters (RA-L), 2025*

---

## Overview

DIVO is a reinforcement learning framework that enables **zero-shot adaptation** to environments with **unseen, dynamically changing obstacles** by:

- Training a **diverse policy** via **random obstacle deployment**
- Learning a **state-dependent latent skill sampler**
- Predicting future trajectories with a **motion predictor**
- Filtering unsafe actions during evaluation


## Setup

Install the required dependencies:

```bash
conda create -n divo python=3.8
conda activate divo
pip install -r requirements.txt
```

---

## Training

### 1. Train Policy with Random Obstacles

```bash
python train.py --config-dir=config/pusht --config-name=train_td3_mujoco_obstacle.yaml
```

### 2. Generate Dataset and Train Latent Skill Sampler (Flow Matching)

```bash
python train.py --config-dir=config/sampler --config-name=train_flow_sampler.yaml
```

### 3. Train Motion Predictor

```bash
python train.py --config-dir=config/motion_decoder --config-name=train_tcn_motion_decoder.yaml
```

---

## Evaluation (Zero-Shot Adaptation)

After training, run:

```bash
python evaluation.py --config-dir=config/evaluation --config-name=eval.yaml
```

This evaluates the trained policy in **unseen environments** with **static and dynamic obstacles**, using skill resampling and motion prediction to filter unsafe actions.

---

## Citation

If you use this work, please cite:

```bibtex
@ARTICLE{10847909,
  author={Choi, Seokjin and Lee, Yonghyeon and Kim, Seungyeon and Park, Che-Sang and Hwang, Himchan and Park, Frank C.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Diverse Policy Learning via Random Obstacle Deployment for Zero-Shot Adaptation}, 
  year={2025},
  volume={10},
  number={3},
  pages={2510-2517},
  keywords={Training;Mutual information;Trajectory;Reinforcement learning;Artificial intelligence;Technological innovation;Navigation;Information filters;Dynamics;Stochastic processes;Reinforcement learning;motion and path planning},
  doi={10.1109/LRA.2025.3532162}}
```

---

## Acknowledgements

This work was supported by IITP, KIAT, SRRC NRF, SNU-AIIS, SNU-IPAI, SNU-IAMD, BK21+, KIAS, and Microsoft Research.
