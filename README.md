# Vision‑Language Model Laboratory + Duckietown Simulator Snapshot

This repository aggregates two pieces that can be used together or separately:
1. `prompt_engineering_lab/` — a lightweight Vision‑Language Model (VLM) experimentation lab for running multiple open‑source VLMs on Duckietown‑style images (interactive offline batch tester + optional live/manual control scripts).
2. `gym-duckietown/` — a vendored snapshot of the Duckietown simulator (refer to its own README for full installation if you need the environment for live control).

The current VLM workflow emphasizes producing consistent, structured JSON describing a driving scene. A curated labeled set lives in `prompt_engineering_lab/examples_to_use/` (PNG images + `vlms_json_labels.json`).

## Contents at a Glance
| Path | Purpose |
|------|---------|
| `prompt_engineering_lab/` | Offline VLM tester, config template, sample images, labels JSON, unified live simulator + VLM runners |
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
- Live manual control now uses a single generic `simulator.py` with pluggable model runners in `vlm_runners.py` (see section below).
- Extra live control requires a functional graphics stack + Duckietown simulator dependencies (see `gym-duckietown/README.md`). Use the vendored snapshot—avoid mixing with a newer upstream unless you intentionally upgrade.

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
## 6. Project Structure (Expanded)
```
vlm_laboratories/
├── README.md                       # Main overview (this file)
│    
├── gym-duckietown/                 # Vendored Duckietown simulator snapshot
│   ├── benchmark.py
│   ├── manual_control.py
│   └── ...                         # See its own README for full details
├── prompt_engineering_lab/
│   ├── vlm_image_tester.py         # Offline batch tester CLI
│   ├── vlm_image_config_example.yml# Example YAML config with prompt schema
│   ├── examples_to_use/            # Sample images + labels JSON (ground truth)
│   ├── vlm_image_results/          # Per-image & per-model output JSON/text (generated)
│   ├── live_vlm_test/              # Live simulator + VLM integration
│   │   ├── simulator.py            # Unified manual control runtime
│   │   ├── vlm_runners.py          # Model runner implementations (Phi/Qwen/Tiny etc.)
│   │   ├── phi_vlm_manual_control.py
│   │   ├── qwen_vlm_manual_control.py
│   │   └── tiny_vlm_manual_control.py
│   └── requirements.txt            # Lab-specific Python dependencies
|	└── README.md                   # Detailed description of this lab usage
├── rag_database_lab/               # Retrieval-Augmented Generation (RAG) rule applicability lab
│   ├── base_instruction.txt        # Single-rule evaluation prompt template
│   ├── batch_instruction.txt       # Multi-rule ranking prompt template
│   ├── clip_rule_retrieval.py      # CLIP embedding & ranking utilities
│   ├── qwen_runner_demo.ipynb      # Notebook comparing VLM vs CLIP
│   ├── rules.json                  # Traffic/behavior rule catalog
│   └── example_images/             # Scene images used for scoring/retrieval
└── prompt_engineering_lab/
```

## 7. Live Manual Control (Unified Simulator)

The previous per‑model manual control scripts have been refactored into a single generic simulator and shared model runners.

Key components:
- `live_vlm_test/simulator.py` — launches Duckietown manual control window; press ENTER to run the chosen VLM on the current frame.
- `live_vlm_test/vlm_runners.py` — defines `VLMRunnerPhi`, `VLMRunnerQwen`, `VLMRunnerTiny` and a `get_vlm_runner()` factory.
- Legacy wrappers (`phi_vlm_manual_control.py`, etc.) still work and auto‑inject their model flag.

### Usage
Activate your virtual environment and ensure simulator dependencies are installed (see earlier sections), then:

```bash
python prompt_engineering_lab/live_vlm_test/simulator.py --vlm-model phi
python prompt_engineering_lab/live_vlm_test/simulator.py --vlm-model qwen
python prompt_engineering_lab/live_vlm_test/simulator.py --vlm-model tiny
```

Additional useful flags (from `simulator.py`):
- `--map-name udem1` choose map
- `--frame-skip 1` frame skip
- `--max-new-tokens 128` generation budget
- `--vlm-log-dir vlm_logs` save frame + text outputs (timestamped)

### Why the Refactor?
Consolidation reduces duplicated keyboard loop / logging logic and makes adding new VLMs as simple as extending `RUNNER_MAP` in `vlm_runners.py`.

### Adding a New Model
1. Implement a class with a `generate(image: PIL.Image, user_text: str, max_new_tokens: int) -> str` method.
2. Register it in `RUNNER_MAP`.
3. Run: `python simulator.py --vlm-model yourkey`.

---

## 8. Troubleshooting
- No images processed: Ensure PNG/JPG files actually exist in `prompt_engineering_lab/examples_to_use/`.
- Empty / partial outputs: First run still downloading weights; wait for completion or check for Hugging Face auth / rate limit messages.
- Headless server: Set environment variables and use `--no-preview` to disable the interactive window.

**Linux/macOS:**
```bash
export MPLBACKEND=Agg
export DISPLAY=:1
unset PYGLET_HEADLESS
```

**Windows:**
```cmd
set MPLBACKEND=Agg
```

- Very slow first inference: Weight + tokenizer load; subsequent runs are faster due to caching.



---

## 9. RAG Database Lab: VLM and CLIP Rule Applicability Benchmark

The `rag_database_lab/` directory contains a self-contained exercise for evaluating and comparing Vision-Language Models (VLMs) and CLIP-based retrieval on the task of rule applicability in simulated driving scenes.

### Purpose
This lab demonstrates retrieval-augmented generation (RAG) for traffic/behavior rule assessment. It enables you to:
- Prompt VLMs to score how well each rule applies to a given simulation image (iterative and batch modes)
- Use CLIP to retrieve and rank rules by visual-text similarity
- Compare and analyze agreement between VLM and CLIP outputs

### Important note
Remember to execute all scripts from jupyter nootebook inside the virtual environment used previously.


### Contents
| File/Folder                | Purpose |
|---------------------------|---------|
| `base_instruction.txt`    | Prompt template for single-rule VLM evaluation (outputs JSON with rule_id, score, and justification) |
| `batch_instruction.txt`   | Prompt template for batch VLM evaluation (outputs JSON array of rule_ids ranked by applicability) |
| `clip_rule_retrieval.py`  | Utilities for CLIP-based rule embedding, ranking, and metrics computation |
| `example_images/`         | Sample PNG images for evaluation |
| `qwen_runner_demo.ipynb`  | Main notebook: runs VLM and CLIP evaluations, computes agreement metrics |
| `rules.json`              | List of traffic/behavior rules (id, rule_text) |

### Quickstart
1. Activate your Python environment and install dependencies (see above).
2. Open `rag_database_lab/qwen_runner_demo.ipynb` in Jupyter or VS Code.
3. Run all cells to:
	- Load rules and images
	- Run VLM iterative and batch evaluations
	- Run CLIP-based retrieval
	- Compare and visualize results

#### Example Outputs
- Per-rule JSON scores from VLM (iterative)
- Ranked rule lists from VLM (batch) and CLIP
- Correlation and overlap metrics between VLM and CLIP

This lab is ideal for benchmarking, prompt engineering, and exploring the strengths/weaknesses of different vision-language approaches for structured scene understanding.

---
Happy experimenting! Adapt the schema to whatever downstream planning or policy tasks you have in mind.
