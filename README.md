# Vision‑Language Model Laboratory + Duckietown Simulator Snapshot

This repository aggregates two pieces that can be used together or separately:
1. `prompt_engineering_lab/` — a lightweight Vision‑Language Model (VLM) experimentation lab for running multiple open‑source VLMs on Duckietown‑style images (interactive offline batch tester + optional live/manual control scripts).
2. `gym-duckietown/` — a vendored snapshot of the Duckietown simulator (refer to its own README for full installation if you need the environment for live control).

The current VLM workflow emphasizes producing consistent, structured JSON describing a driving scene. A curated labeled set lives in `prompt_engineering_lab/examples_to_use/` (PNG images + `vlms_json_labels.json`).

## Contents at a Glance
| Path | Purpose |
|------|---------|
| `prompt_engineering_lab/` | Offline VLM tester, config template, sample images, labels JSON, live scripts |
| `gym-duckietown/` | Duckietown simulator snapshot (only needed for live control) |

You can run the offline tester without installing or launching the simulator.

## 1. Python 3.8 Installation & Environment Setup

### Download and Install Python 3.8
First, ensure you have Python 3.8 installed on your system:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install python3.8 python3.8-venv python3.8-devel
# or for older versions: sudo yum install python38 python38-venv python38-devel
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.8
```

**Windows:**
Download Python 3.8 from [python.org](https://www.python.org/downloads/release/python-3810/) and install it. Make sure to check "Add Python to PATH" during installation.

### Create Virtual Environment
Duckietown simulator is using older versions of some libraries as well as Python. To launch everything easily every time virtual environment setup is needed.

**Linux/macOS:**
```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Windows (Command Prompt):**
```cmd
python3.8 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```


To activate in the future:

**Linux/macOS:**
```bash
source .venv/bin/activate
python --version  # Should show Python 3.8.x
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate
python --version  # Should show Python 3.8.x
```


**Note:** If `python3.8` command is not found on Windows, you may need to use `py -3.8` instead:
```cmd
py -3.8 -m venv .venv
```

## 2. Install Dependencies

### Install Duckietown Simulator (Required for Live Control)
Install the gym-duckietown package in editable mode:

**Linux/macOS:**
```bash
pip3 install -e gym-duckietown
```

**Windows:**
```cmd
pip install -e gym-duckietown
```

### Install VLM Lab Requirements
Install everything needed for the offline tester:

**All platforms:**
```bash
pip install -r prompt_engineering_lab/requirements.txt
```
Libraries should be installed now. If something is missing istall it manually using `pip`.

Notes:
- First use of each VLM pulls model weights from Hugging Face (one‑time download).
- Extra live control simulator additionally requires a functional graphics stack + Duckietown simulator dependencies (see `gym-duckietown/README.md`). If instalation is needed, use provided version - do not clone newest one.

## 3. Labeled Example Images
Folder: `prompt_engineering_lab/examples_to_use/`

Contains:
- `*.png` sample perception frames
- `vlms_json_labels.json` with a schema:
	- image (str)
	- obstacle_type (None|Vehicle|Pedestrian|Static)
	- lane_position (Left|Center|Right|Off Track)
	- pedestrian_presence (None|Near)
	- collision_warning (true|false)
	- intersection_ahead (true|false)
	- is_night_time (true|false)
	- road_type (Urban|Rural|Under Construction)
	- obstacle_distance (Near|Moderate|Far|None)

Important: The `vlm_image_tester.py` script loads images directly from `examples_to_use/`. 
## 4. Quick Start (30‑Second Demo)

**Linux/macOS:**
```bash
source .venv/bin/activate

# Run all supported models with the provided YAML config
python3 prompt_engineering_lab/vlm_image_tester.py \
	--config prompt_engineering_lab/vlm_image_config_example.yml
```

**Windows:**
```cmd
.venv\Scripts\activate

python prompt_engineering_lab/vlm_image_tester.py --config prompt_engineering_lab/vlm_image_config_example.yml
```
Outputs: per‑image & per‑model text files under `prompt_engineering_lab/vlm_image_results/` plus a copy of each image.

## 5. Using / Adapting the Prompt
The default config file (`vlm_image_config_example.yml`) ships with some example prompts such as:
```yaml
prompt: |
	You are a driving scene understanding assistant. Analyze this single front‑facing image from a robot simulator.
	OUTPUT ONLY one JSON object (no prose, no markdown). Use these exact keys:
	{
		"obstacle_type": "None|Vehicle|Pedestrian|Static",
		"lane_position": "Left|Center|Right|Off Track",
		"pedestrian_presence": "None|Near",
		"collision_warning": true|false,
		"intersection_ahead": true|false,
		"is_night_time": true|false,
		"road_type": "Urban|Rural|Under Construction",
		"obstacle_distance": "Near|Moderate|Far|None",
	}
	Rules: Output booleans as true/false (not strings). If something is unknown choose the closest valid value. Output ONLY the JSON.
```
There is a special field and all the suggestions how to insert own prompt.
## 6. Project Structure (Abridged)
```
vlm_laboratories/
├── .venv/
├── README.md
├── gym-duckietown/
└── prompt_engineering_lab/
		├── vlm_image_tester.py          # offline tester CLI (loads from examples_to_use/)
		├── vlm_image_config_example.yml
		├── examples_to_use/             # sample images + labels JSON (consumed directly)
		├── vlm_image_results/           # (created) model outputs
		└── live_vlm_test/               # optional live scripts
```

## 7. Troubleshooting
- No images processed: Ensure PNG/JPG files actually exist in `prompt_engineering_lab/examples_to_use/`.
- Empty / partial outputs: First run still downloading weights; wait for completion or check for Hugging Face auth / rate limit messages.
- Headless server: Set environment variables and use `--no-preview` to disable the interactive window.

**Linux/macOS:**
```bash
export MPLBACKEND=Agg
export DISPLAY=:0
```

**Windows:**
```cmd
set MPLBACKEND=Agg
```

- Very slow first inference: Weight + tokenizer load; subsequent runs are faster due to caching.

## 8. Next Steps & Ideas
- See `prompt_engineering_lab/README.md` for detailed flags, configuration, and live control usage.

---
Happy experimenting! Adapt the schema to whatever downstream planning or policy tasks you have in mind.
