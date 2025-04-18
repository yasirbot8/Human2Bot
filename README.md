# Human2Bot: Learning Zero-Shot Reward Functions for Robotic Manipulation from Human Demonstrations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasirbot8/Human2Bot)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://yasirbot8.github.io/Human2Bot)

**Human2Bot (H2B)** is a framework for zero-shot reward learning in robotic manipulation tasks. It learns reward functions directly from human video demonstrations, without requiring any robot-specific data. By leveraging multi-scale inter-frame attention and domain-augmented visual embeddings, Human2Bot enables robots to generalize across tasks and environments—even in previously unseen scenarios.

---

## 🚀 Quick Installation

1. **Clone the repository**
```bash
git clone https://github.com/yasirbot8/Human2Bot.git
cd Human2Bot
```
2. **Create and Activate Conda Environment**
```bash
conda env create -f conda_env.yml
conda activate Human2Bot
```
3. **Install Python Requirements**
```bash
pip install -r requirements.txt
```
4. **Install Simulation Environments**

    Follow the instructions to install the TableTop environment and related Meta-World dependencies [here](https://github.com/anniesch/dvd).

## 📦 Dataset (Pre-Computed Embeddings)
Download the pre-computed embeddings from the Something-Something-V2 with 50% augmentation [here](https://drive.google.com/file/d/14HjUvqodZaO5DEb4Qt-TEoPNpe7sa-P1/view?usp=drive_link).

## 🏋️‍♂️ Training Human2Bot

To train the reward classifier, run:

```bash
python main.py
```

## 🧠 Planning with Human2Bot

To generate task plans using the learned reward function, run:

```bash
python plan.py
```
## 📖 Citation

If you use this work, please cite the following paper:

**Salam, Y.**, **Li, Y.**, **Herzog, J.**, et al. *Human2bot: Learning Zero-Shot Reward Functions for Robotic Manipulation from Human Demonstrations*. Auton Robot, 49(10), 2025. [https://doi.org/10.1007/s10514-025-10193-9](https://doi.org/10.1007/s10514-025-10193-9)


