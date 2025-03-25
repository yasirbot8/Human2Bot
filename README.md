# Human2Bot: Zero-Shot Reward Learning for Robotic Manipulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasirbot8/Human2Bot)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://yasirbot8.github.io/Human2Bot)


## Quick Installation

```bash
# Clone repository
git clone https://github.com/yasirbot8/Human2Bot.git
cd Human2Bot

# Create conda environment
conda env create -f conda_env.yml
conda activate Human2Bot

# Install requirements
pip install -r requirements.txt

# Install Meta-World dependency
git clone https://github.com/anniesch/dvd && cd dvd && pip install -e . && cd ..

### Download Pre-Computed Embeddings

1. **Download the pre-processed embeddings**:
   - [Something-Something-V2 with 50% augmentation (Main Dataset)](https://drive.google.com/drive/folders/YOUR_MAIN_EMBEDDING_FOLDER)
   - [Task-Specific Embeddings (Optional)](https://drive.google.com/drive/folders/YOUR_TASK_SPECIFIC_FOLDER)
