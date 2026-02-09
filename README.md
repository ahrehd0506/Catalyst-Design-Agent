# Catalyst-Design-Agent

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
