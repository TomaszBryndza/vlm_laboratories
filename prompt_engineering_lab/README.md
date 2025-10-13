# Duckietown VLM Image Tester & Live Control (Structured JSON Labeling Focus)

Compact lab to evaluate multiple open‑source Vision‑Language Models (VLMs) on Duckietown‑style driving scenes. It provides: (1) an offline image tester with interactive preview + YAML‑driven prompts, and (2) optional live "manual control" scripts that run VLMs on frames from a Duckietown simulator. The primary goal is consistent, structured JSON describing each scene.

## Table of Contents
- [Duckietown VLM Image Tester \& Live Control (Structured JSON Labeling Focus)](#duckietown-vlm-image-tester--live-control-structured-json-labeling-focus)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Offline image tester (recommended starting point)](#offline-image-tester-recommended-starting-point)
    - [Live manual control (advanced)](#live-manual-control-advanced)
  - [Configuration](#configuration)
  - [Examples \& Labels](#examples--labels)
  - [CLI Flags Quick Reference](#cli-flags-quick-reference)
  - [Evaluating Outputs vs Labels](#evaluating-outputs-vs-labels)
  - [Technologies Used](#technologies-used)
  - [Acknowledgements](#acknowledgements)

## Overview
This lab compares multiple VLMs on the same image and prompt to explore model behavior for autonomous‑driving‑style decisions. The offline tester auto‑discovers images from a local folder, runs one or all supported VLMs, shows an interactive side‑by‑side preview (image + live results), prints results to the console, and optionally saves outputs per image.

The live control scripts mirror the classic Duckietown manual control app and let you press ENTER to run a VLM on the current camera frame (for description and suggested action).

## Features
- Run one or all supported VLMs on each image with a single CLI.
- Interactive preview with live‑updating results, plus console mirroring.
- YAML‑configurable prompts and mode selection (single/all, which VLM).
- Automatic discovery of images from a local folder.
- Per‑image, per‑model output files (text) and optional saved image copies.
Three VLM backends supported out of the box:
  - microsoft/Phi‑3.5‑vision‑instruct
  - anananan116/TinyVLM
  - Qwen/Qwen2‑VL‑2B‑Instruct
- Optional live “manual control” scripts for on‑demand VLM inference on simulator frames.

Structured label objective (current lab focus): prompt the models to output a JSON object capturing scene attributes (obstacles, lane position, etc.) matching the schema in `examples_to_use/vlms_json_labels.json`.

## Project Structure

📦 prompt_engineering_lab/
- `README.md` – This documentation
- `requirements.txt` – Core Python packages (torch, transformers, torchvision, pillow, pyyaml, matplotlib)
- `vlm_image_tester.py` – Main CLI (loads images from `examples_to_use/`)
- `vlm_image_config_example.yml` – Example YAML config (note: filename contains `_example` not `.example`)
- `live_vlm_test/` – Live/manual control scripts
  - `phi_vlm_manual_control.py`
  - `tiny_vlm_manual_control.py`
  - `qwen_vlm_manual_control.py`
- `examples_to_use/` – Sample images + `vlms_json_labels.json` rich label schema (consumed directly by the tester)

## Installation

Requirements:
- Python 3.8 (CPU‑only is fine; GPU optional if available). (Should be installed in .venv also)
- pip/virtualenv (recommended)

1) Activate the local virtual environment from the repository root (recommended):

**Linux/macOS:**
```bash
source ../.venv/bin/activate  # from prompt_engineering_lab/ directory
python --version # check if version = 3.8.x
```

**Windows (Command Prompt):**
```cmd
..\.venv\Scripts\activate
python --version 
```
Check if Python version is 3.8.x.
1) Install core dependencies for the offline tester into the active .venv (the extra utility libs are already declared in `requirements.txt`):

**All platforms:**
```bash
pip install -r requirements.txt
```

3) (Optional) For live Duckietown manual control scripts, you’ll also need a working Duckietown/gym environment with a display (or Xvfb). That setup is outside the scope of this lab, but if you are using it inside the Duckietown repository, follow their installation docs first. No need to clone gym-duckietown repo - it is provided inside this repository.

## Usage

### Offline image tester (recommended starting point)
Images are automatically loaded from `examples_to_use/`. Add or replace PNG/JPG/JPEG/BMP/WebP files there.

-- Run all models with the example YAML prompt:

**Linux/macOS:**
```bash
python3 vlm_image_tester.py --mode all --prompt "Describe the scene and propose a safe action."
```

**Windows:**
```cmd
python vlm_image_tester.py --mode all --prompt "Describe the scene and propose a safe action."
```

- Run a single model with a quick inline prompt:

**Linux/macOS:**
```bash
python3 vlm_image_tester.py --mode single --vlm tiny --prompt "Describe the scene and propose a safe action."
```

**Windows:**
```cmd
python vlm_image_tester.py --mode single --vlm tiny --prompt "Describe the scene and propose a safe action."
```

For use in the lab there is `vlm_image_config_control.yml` prepared. Inside this file user can write a prompt and choose script mode. It is more recommended, especially for the long prompts. Script can be launched then, with only config flag.

**Linux/macOS:**
```bash
python3 vlm_image_tester.py --config vlm_image_config_example.yml
```

**Windows:**
```cmd
python vlm_image_tester.py --config vlm_image_config_example.yml
```
Other additional flags of interest for console launch:
- `--mode {single,all}` choose one model or all
- `--vlm {phi,tiny,qwen}` which model in single mode
- `--prompt "..."` inline prompt (overrides config)
- `--config path.yml` YAML with `mode`, `vlm`, and `prompt`
- `--preview/--no-preview` show or disable the interactive window (default: preview on)
- `--save-dir DIR` where results are written (default: `vlm_image_results/`)
- `--max-new-tokens N` generation length budget (default: 128)

Outputs: For each `image.png` and model `tiny`, you’ll get `vlm_image_results/image_tiny.txt` containing the prompt and the model’s output. A copy of the input image is also saved once per image name. Use these to compare against labels if you adapt an evaluation script.

### Live manual control (advanced)
These scripts open a Duckietown manual control window and let you press ENTER to run a VLM on the current frame:

**Linux/macOS:**
- Phi: `python3 live_vlm_test/phi_vlm_manual_control.py`
- TinyVLM: `python3 live_vlm_test/tiny_vlm_manual_control.py`
- Qwen: `python3 live_vlm_test/qwen_vlm_manual_control.py`

**Windows:**
- Phi: `python live_vlm_test/phi_vlm_manual_control.py`
- TinyVLM: `python live_vlm_test/tiny_vlm_manual_control.py`
- Qwen: `python live_vlm_test/qwen_vlm_manual_control.py`

You may need additional dependencies (Gym, Pyglet, Duckietown assets) and a working display. 

## Configuration
Use a YAML file to control the tester’s prompt and which model(s) to run. Keys:
- `mode`: `single` or `all`
- `vlm`: `phi`, `tiny`, or `qwen` (used when `mode: single`)
- `prompt`: instruction text shown to the VLMs

Example (`vlm_image_config_example.yml`):
```yaml
mode: all
vlm: tiny
prompt: |
  You are a driving assistant. Analyze a single front-facing road image from the simulation environment of a robot. OUTPUT ONLY a valid JSON object (no extra text, no markdown, no code fences). Return exactly these fields:

{
    "obstacle_type": "None|Vehicle|Pedestrian|Static",
    "lane_position": "Left|Center|Right|Off Track",
    "pedestrian_presence": "None|Near",
    "collision_warning": true|false,
    "intersection_ahead": true|false,
    "is_night_time": true|false,
    "road_type": "Urban|Rural|Under Construction",
    "obstacle_distance": "Near|Moderate|Far|None"
  }

  Rules:
  Choose answer only from those provided.
  Treat yellow duck figures as pedestrians.
  Do not echo the instructions. Output only the JSON object.

```

## Examples & Labels
Folder: `examples_to_use/`

Labels file: `vlms_json_labels.json` (schema). Each entry includes:
- image (string filename)
- obstacle_type (None|Vehicle|Pedestrian|Static)
- lane_position (Left|Center|Right|Off Track)
- pedestrian_presence (None|Near)
- collision_warning (boolean)
- intersection_ahead (boolean)
- is_night_time (boolean)
- road_type (Urban|Rural|Under Construction)
- obstacle_distance (Near|Moderate|Far|None)

Workflow:
1. Place / verify images in `examples_to_use/`.
2. Adjust `vlm_image_config_example.yml` prompt (see below) to request exactly the above fields.
3. Run tester; collect outputs in `vlm_image_results/`.
4. (Optional) Evaluate model JSON vs labels (see evaluation section below).



## CLI Flags Quick Reference

| Flag | Values | Purpose |
|------|--------|---------|
| `--mode` | `single` | Run exactly one specified VLM |
|        | `all` | Run every supported VLM sequentially / concurrently (UI) |
| `--vlm` | `phi`, `tiny`, `qwen` | Which model in single mode |
| `--prompt` | string | Inline prompt (overrides YAML) |
| `--config` | path.yml | YAML with `mode`, `vlm`, `prompt` |
| `--max-new-tokens` | int (default 128) | Generation length budget |
| `--preview / --no-preview` | bool | Enable/disable interactive matplotlib window |
| `--save-dir` | path (default `vlm_image_results`) | Output directory (empty string disables saving) |
| `--color / --no-color` | bool | Force ANSI color on/off |

## Evaluating Outputs vs Labels
There is currently no baked‑in evaluator. To add one quickly:
1. Parse `vlms_json_labels.json` into a dict keyed by image filename.
2. For each model output file in `vlm_image_results/`, extract the last JSON object (robustly strip preamble).
3. Validate JSON: ensure all required keys exist; coerce casing if needed.
4. Compute per‑field accuracy (exact match) and overall exact‑record accuracy.
5. (Optional) Add relaxed metrics (e.g., treat `Vehicle` vs `Static` as partial credit.

Potential future enhancements:
- Weighted scoring fields in the JSON schema (safety‑critical fields like `emergency_braking`).
- Confusion matrix per categorical field.
- JSON schema validation step to fail fast on malformed outputs.

## Technologies Used
- Python 3.8+
- PyTorch (`torch`), Transformers (`transformers`), TorchVision
- Pillow (PIL), Matplotlib (interactive preview)
- Optional (live control): Gym, Pyglet, and Duckietown environment/assets
- Hugging Face models: Phi‑3.5‑Vision‑Instruct, TinyVLM, Qwen2‑VL‑2B‑Instruct


## Acknowledgements
- Duckietown project and community for the simulator and tasks
- Hugging Face and model authors: Microsoft (Phi‑3.5‑Vision), Alibaba (Qwen2‑VL), and TinyVLM contributors
- Open‑source libraries: PyTorch, Transformers, Pillow, Matplotlib, Gym, Pyglet
