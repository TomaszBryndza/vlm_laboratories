# Vision‑Language Model Laboratory + Duckietown Simulator Snapshot

This repository aggregates two pieces that can be used together or separately:
1. `prompt_engineering_lab/` — a lightweight Vision‑Language Model (VLM) experimentation lab for running multiple open‑source VLMs on Duckietown‑style images (interactive offline batch tester + optional live/manual control scripts).
2. `gym-duckietown/` — a vendored snapshot of the Duckietown simulator (refer to its own README for full installation if you need the environment for live control).

The current VLM workflow emphasizes producing consistent, structured JSON describing a driving scene. A curated labeled set lives in `prompt_engineering_lab/examples_to_use/` (PNG images + `vlms_json_labels.json`).

## Contents at a Glance
| Path | Purpose |
|------|---------|
| `.venv/` | Local Python virtual environment (create/activate before running) |
| `prompt_engineering_lab/` | Offline VLM tester, config template, sample images, labels JSON, live scripts |
| `gym-duckietown/` | Duckietown simulator snapshot (only needed for live control) |

You can run the offline tester without installing or launching the simulator.

## 1. Environment
Activate (if already created):
```bash
source .venv/bin/activate
python --version
```
Create if missing:
```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install Dependencies
Install everything needed for the offline tester (the extra pillow/pyyaml/matplotlib lines are already present in `requirements.txt`):

```bash
pip install -r prompt_engineering_lab/requirements.txt
```
Libraries should be included in the virtual environment. If something is missing istall it manually.

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
```bash
source .venv/bin/activate

# Run all supported models with the provided YAML config
python3 prompt_engineering_lab/vlm_image_tester.py \
	--config prompt_engineering_lab/vlm_image_config_example.yml
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
- Headless server: Set `MPLBACKEND=Agg` and `DISPLAY=:0`/`DISPLAY=:1` or run with `--no-preview` to disable the interactive window.
- Very slow first inference: Weight + tokenizer load; subsequent runs are faster due to caching.

## 8. Next Steps & Ideas
- See `prompt_engineering_lab/README.md` for detailed flags, configuration, and live control usage.

---
Happy experimenting! Adapt the schema to whatever downstream planning or policy tasks you have in mind.
# vlm_laboratories
