# MAESTRO: Mulit-Agent-based Electrocatalyst Search Through Reasoning and Optimization 

## Installation

Run the following commands to set up:

```bash
conda update conda
conda env create -f env.yml
```
Activate the Conda environment with `conda activate catagent`

## Run

### 1. Configure the settings
Before running, modify the configuration file to paths, options, and parameters:

```bash
vi config/config.yml
```
### 2. Set API KEY
Inside agent.py, locate the following lines:

```bash
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"
```

Replace "YOUR_API_KEY_HERE" with your own API key, or set API keys via environment variables.

### 3. Run the agent

```bash
python agent.py
```

## Citation

```
@article{mok2026reasoning,
  title={Reasoning-Driven Design of Single Atom Catalysts via a Multi-Agent Large Language Model Framework},
  author={Mok, Dong Hyeon and Back, Seoin and Fung, Victor and Hu, Guoxiang},
  journal={arXiv preprint arXiv:2602.21533},
  year={2026}
}
```
