#!/usr/bin/env python3
"""
Compare multiple Vision-Language Models (VLMs) on the same image and prompt.

Two modes are supported:
- single: user selects one VLM to run on the provided image(s)
- all:    runs all available VLMs on each provided image

The script previews the image being processed and prints the generated output.
Images are automatically loaded from the "example_images/" folder next to this script.
Optionally, results are saved into a results directory next to the script.

Usage examples:
  # Single model
    python vlm_image_tester.py --mode single --vlm tiny --prompt "Describe the scene."

  # All models
    python vlm_image_tester.py --mode all --prompt "What should the car do?"

# Using a YAML config file for the prompt (images are always read from examples_to_use/):
    # config.yml
    #   prompt: |
    #     Describe the image in detail and propose a safe driving action.
    #
    python vlm_image_tester.py --config config.yml

Requirements:
    - Shared VLM runners implemented in `live_vlm_test/vlm_runners.py` (factory via get_vlm_runner).
        Legacy manual control scripts no longer contain runner classes after refactor.
    - torch, transformers, Pillow
    - matplotlib (optional, used for inline preview; PIL fallback is used otherwise)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple
import threading
import textwrap
import traceback
from matplotlib.widgets import Button  

import torch
from PIL import Image
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Make sure we can import sibling modules when running the script directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# VLM runner classes are imported lazily inside make_vlm_runner() to avoid
# heavy side effects at import time and to speed up CLI parsing.


# ---- Console color helpers (ANSI) ----
USE_COLOR: bool = False  # set in main based on --color/--no-color or TTY
RESET = "\033[0m"
FG_RED = "\033[31m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_CYAN = "\033[36m"

def _supports_color() -> bool:
    try:
        return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    except Exception:
        return False

def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if USE_COLOR else text


def show_interactive_preview(
    image: Image.Image,
    img_path: str,
    prompt: str,
    vlm_kinds: List[str],
    runners: Dict[str, object],
    device: torch.device,
    max_new_tokens: int,
    save_dir: Optional[str],
) -> None:
    """Show an interactive preview with the image on the left and text on the right.

    While the window is open, run the VLM(s) in background threads. The right panel
    displays the prompt and live-updating generation results for each model.
    The window remains until the user closes it.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        # Fallback if matplotlib is unavailable: run sequentially and use PIL preview
        image.show(title=os.path.basename(img_path))
        # Run sequentially as a fallback
        for kind in vlm_kinds:
            if kind not in runners:
                runners[kind] = make_vlm_runner(kind, device)
            ok, text = run_vlm(runners[kind], image, prompt, max_new_tokens)
            print(f"\n[{kind.upper()} Output]\n{text}")
        print(c("\n[ALL DONE] Finished generating for: ", FG_GREEN) + os.path.basename(img_path))
        return

    # Save the image once into save_dir for convenience
    if save_dir is not None:
        try:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_img = os.path.join(save_dir, f"{stem}.png")
            if not os.path.exists(out_img):
                image.save(out_img)
        except Exception as e:
            print(f"[SAVE-ERROR] Could not save image copy: {e}")

    # Shared state for threads
    statuses: Dict[str, str] = {k: "queued" for k in vlm_kinds}
    outputs: Dict[str, str] = {k: "" for k in vlm_kinds}
    lock = threading.Lock()

    # Pre-initialize models sequentially to avoid race conditions on first load
    for kind in vlm_kinds:
        with lock:
            statuses[kind] = "loading"
            print(c(f"\n[INIT] Loading {kind} model on {device} ...", FG_CYAN))
        try:
            if kind not in runners:
                t0 = time.time()
                runners[kind] = make_vlm_runner(kind, device)
                with lock:
                    print(c(f"[INIT] Loaded {kind} in {time.time() - t0:.1f}s", FG_GREEN))
            with lock:
                statuses[kind] = "queued"
        except Exception as e:
            tb = traceback.format_exc()
            with lock:
                outputs[kind] = f"ERROR during init: {e}\n{tb}"
                statuses[kind] = "error"
            print(c(f"[INIT-ERROR] Could not initialize '{kind}': {e}", FG_RED))
            print(tb)

    def worker(kind: str) -> None:
        nonlocal runners
        try:
            with lock:
                statuses[kind] = "generating"
                print(c(f"[{kind.upper()}] Generating ...", FG_YELLOW))
            tries = 0
            last_err_text = ""
            while tries < 2:
                try:
                    with torch.inference_mode():
                        text = runners[kind].generate(image, prompt, max_new_tokens=max_new_tokens)
                    with lock:
                        outputs[kind] = text
                        statuses[kind] = "done"
                        header = kind.upper()
                        print(c(f"\n[{header} Output]", FG_GREEN) + f"\n{text}")
                    break
                except Exception as e:
                    tb = traceback.format_exc()
                    last_err_text = f"ERROR: {e}\n{tb}"
                    tries += 1
                    time.sleep(0.3)
            if tries >= 2:
                with lock:
                    outputs[kind] = last_err_text
                    statuses[kind] = "error"
                print(c(f"\n[{kind.upper()} ERROR]", FG_RED) + f"\n{last_err_text}")
            # Save result if requested
            if save_dir is not None:
                try:
                    stem = os.path.splitext(os.path.basename(img_path))[0]
                    out_txt = os.path.join(save_dir, f"{stem}_{kind}.txt")
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(f"image={img_path}\n")
                        f.write(f"prompt={prompt}\n\n")
                        f.write(outputs[kind])
                except Exception as e:
                    # Log to console but do not interrupt UI
                    print(c(f"[SAVE-ERROR] Could not save results for {kind}: {e}", FG_RED))
        except Exception as e:
            with lock:
                outputs[kind] = f"ERROR: {repr(e)}"
                statuses[kind] = "error"

    # Launch threads only for successfully initialized models
    threads: List[threading.Thread] = []
    for k in vlm_kinds:
        with lock:
            if statuses.get(k) == "error":
                continue
        t = threading.Thread(target=worker, args=(k,), daemon=True)
        threads.append(t)
        t.start()

    # Build figure with image and text panel
    import matplotlib.pyplot as plt  # type: ignore
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [2, 3]},
        num=f"{os.path.basename(img_path)}",
    )
    ax_img, ax_txt = axes
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title(os.path.basename(img_path))

    ax_txt.axis("off")
    ax_txt.set_title("Prompt and Results")

    # Prepare text object for updates
    text_obj = ax_txt.text(
        0.01,
        0.98,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax_txt.transAxes,
        wrap=True,
    )

    # Add a Close button in the bottom-right area of the figure
    # [left, bottom, width, height] in figure coordinates
    btn_ax = fig.add_axes([0.86, 0.02, 0.12, 0.06])
    close_btn = Button(btn_ax, "Close", color="#dddddd", hovercolor="#cccccc")

    def on_close_clicked(event):  # pragma: no cover
        try:
            plt.close(fig)
        except Exception:
            pass

    close_btn.on_clicked(on_close_clicked)

    def build_text() -> str:
        width = 90  # wrap width for text panel
        lines: List[str] = []
        lines.append("Prompt:")
        lines.extend(textwrap.fill(prompt, width=width).splitlines())
        lines.append("")
        all_done = True
        for kind in vlm_kinds:
            with lock:
                status = statuses.get(kind, "queued")
                out = outputs.get(kind, "")
            header = f"[{kind.upper()}]"
            if status in {"queued", "loading"}:
                lines.append(f"{header} Loading model ...")
                all_done = False
            elif status == "generating":
                lines.append(f"{header} Generating ...")
                all_done = False
            elif status == "done":
                lines.append(f"{header} Result:")
                wrapped = textwrap.fill(out, width=width)
                lines.extend(wrapped.splitlines())
            elif status == "error":
                lines.append(f"{header} {out}")
            lines.append("")
        if all_done:
            lines.append("All models finished,  press the window close button to continue.")

        return "\n".join(lines)

    # Event/render loop: update text while window is open
    plt.tight_layout()
    fig.canvas.draw_idle()
    announced_done = False
    while plt.fignum_exists(fig.number):
        text_obj.set_text(build_text())
        fig.canvas.draw_idle()
        # Check completion state
        with lock:
            all_done_now = all(statuses.get(k) in {"done", "error"} for k in vlm_kinds)
        if all_done_now and not announced_done:
            fig.suptitle("All models finished.", color="green", fontsize=11)
            print(c("\n[ALL DONE] Finished generating for: ", FG_GREEN) + os.path.basename(img_path))
            print(c("\nPress the window close button to continue.", FG_CYAN))
            # Update button label to indicate completion
            try:
                close_btn.label.set_text("Close (Done)")
                fig.canvas.draw_idle()
            except Exception:
                pass
            announced_done = True
        plt.pause(0.1)

    # Join threads (they are daemonic; still wait briefly to tidy up)
    for t in threads:
        if t.is_alive():
            t.join(timeout=0.1)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for this script."""
    p = argparse.ArgumentParser(description="Run one or more VLMs on provided images and a prompt.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML file with settings (e.g., prompt, mode, vlm). CLI overrides config.",
    )
    p.add_argument(
        "--mode",
        choices=["single", "all"],
        default=None,
        help="Mode: run a single chosen VLM or all available VLMs (overrides config).",
    )
    p.add_argument(
        "--vlm",
        choices=["phi", "tiny", "qwen"],
        default=None,
        help="Which VLM to run in 'single' mode (ignored in 'all' mode). Overrides config.",
    )
    # Images are always read from example_images/; no CLI images input required.
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Instruction prompt passed to each VLM.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    # Python 3.8-compatible boolean flags for preview
    preview_group = p.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        help="Show a preview of each image before running the VLM(s).",
    )
    preview_group.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Disable preview of images before running the VLM(s).",
    )
    p.set_defaults(preview=True)
    # Colorized console output flags
    color_group = p.add_mutually_exclusive_group()
    color_group.add_argument(
        "--color",
        dest="color",
        action="store_true",
        help="Force enable ANSI color in console logs.",
    )
    color_group.add_argument(
        "--no-color",
        dest="color",
        action="store_false",
        help="Disable ANSI color in console logs.",
    )
    p.set_defaults(color=True)
    p.add_argument(
        "--save-dir",
        type=str,
        default="vlm_image_results",
        help="Directory to save outputs (created if missing). Set empty to disable saving.",
    )
    return p


def load_config(path: Optional[str]) -> Dict:
    """Load YAML config from path, returning an empty dict if none provided.

            Recognized keys:
                - prompt: str
                - mode:   "single" | "all"
                - vlm:    "phi" | "tiny" | "qwen" (used when mode == single)
    """
    if not path:
        return {}
    if yaml is None:
        raise SystemExit("PyYAML is required for --config; install with: pip install pyyaml")
    if not os.path.isfile(path):
        raise SystemExit(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise SystemExit("Config must contain a YAML mapping (key: value)")
        return data


def make_vlm_runner(kind: str, device: torch.device):
    """Factory for VLM runners (post-refactor).

    Delegates to `live_vlm_test.vlm_runners.get_vlm_runner` which centralizes
    the construction logic for all supported models.
    """
    try:
        from live_vlm_test.vlm_runners import get_vlm_runner  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Could not import shared VLM runners module 'vlm_runners'.") from e
    return get_vlm_runner(kind, device)


def ensure_save_dir(save_dir: Optional[str]) -> Optional[str]:
    """Create the save directory if requested and return its path."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    return None


def run_vlm(
    runner,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[bool, str]:
    """Execute generation on the given runner and return (ok, text_or_error)."""
    try:
        with torch.inference_mode():
            out_text = runner.generate(image, prompt, max_new_tokens=max_new_tokens)
        return True, out_text
    except Exception as e:
        return False, f"ERROR: {repr(e)}"


def main() -> None:
    """CLI entry point for testing VLMs on provided images."""
    warnings.filterwarnings("ignore", category=UserWarning)
    args = build_arg_parser().parse_args()

    # Configure color output (default on; --no-color disables)
    global USE_COLOR
    USE_COLOR = bool(args.color)

    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    # Load optional YAML config and compute effective settings (CLI > config > defaults)
    cfg = load_config(args.config)
    default_prompt = (
        "What action should be taken by the vehicle in this situation? "
        "Provide a decision based on the current state of the environment. "
        "Describe what you see, the action to take, and why."
    )
    effective_prompt: str = args.prompt if args.prompt is not None else cfg.get("prompt", default_prompt)

    # Collect images from examples_to_use/ next to this script
    images_dir = os.path.join(SCRIPT_DIR, "examples_to_use")
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not os.path.isdir(images_dir):
        # Create the folder and ask the user to add images
        try:
            os.makedirs(images_dir, exist_ok=True)
        except Exception:
            pass
        raise SystemExit(
            f"Images folder not found. Created: {images_dir}\n"
            "Please add images (png/jpg/jpeg/bmp/webp) to this folder and rerun."
        )
    effective_images: List[str] = []
    for name in sorted(os.listdir(images_dir)):
        path = os.path.join(images_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_ext:
            effective_images.append(path)
    if not effective_images:
        raise SystemExit(
            f"No images found in {images_dir}. Please add image files (png/jpg/jpeg/bmp/webp) and rerun."
        )

    # Decide which models to run (CLI overrides config; default to 'single' if neither provided)
    effective_mode: str = args.mode if args.mode is not None else cfg.get("mode", "single")
    effective_vlm: Optional[str] = args.vlm if args.vlm is not None else cfg.get("vlm")

    if effective_mode not in {"single", "all"}:
        raise SystemExit("mode must be 'single' or 'all' (via --mode or config)")

    if effective_mode == "single":
        if effective_vlm not in {"phi", "tiny", "qwen"}:
            raise SystemExit("vlm must be one of {phi,tiny,qwen} when mode=='single' (via --vlm or config)")
        vlm_kinds = [effective_vlm]
    else:
        vlm_kinds = ["phi", "tiny", "qwen"]

    save_dir = ensure_save_dir(args.save_dir)

    # Initialize runners lazily (first use) to reduce upfront cost
    runners: Dict[str, object] = {}

    for img_path in effective_images:
        if not os.path.isfile(img_path):
            print(f"[WARN] Skipping non-existent file: {img_path}")
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open image {img_path}: {e}")
            continue

        title = f"Preview: {os.path.basename(img_path)}"
        if args.preview:
            print(c("\n======== Running VLM(s) on: ", FG_CYAN) + f"{img_path}" + c(" ========", FG_CYAN))
            print(c("Prompt:", FG_YELLOW), effective_prompt)
            show_interactive_preview(
                image=image,
                img_path=img_path,
                prompt=effective_prompt,
                vlm_kinds=vlm_kinds,
                runners=runners,
                device=device,
                max_new_tokens=args.max_new_tokens,
                save_dir=save_dir,
            )
            # After closing the window, continue to next image
            continue

        # If preview was disabled, run sequentially without UI and print outputs
        print(c("\n======== Running VLM(s) on: ", FG_CYAN) + f"{img_path}" + c(" ========", FG_CYAN))
        print(c("Prompt:", FG_YELLOW), effective_prompt)

        for kind in vlm_kinds:
            # Instantiate on first use
            if kind not in runners:
                t0 = time.time()
                print(c(f"\n[INIT] Loading {kind} model on {device} ...", FG_CYAN))
                try:
                    runners[kind] = make_vlm_runner(kind, device)
                    print(c(f"[INIT] Loaded {kind} in {time.time() - t0:.1f}s", FG_GREEN))
                except Exception as e:
                    print(c(f"[INIT-ERROR] Could not initialize '{kind}': {e}", FG_RED))
                    continue

            ok, text = run_vlm(runners[kind], image, effective_prompt, args.max_new_tokens)
            header = kind.upper()
            print(c(f"\n[{header} Output]", FG_GREEN) + f"\n{text}")

            # Optionally save outputs per image+model
            if save_dir is not None:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                out_txt = os.path.join(save_dir, f"{stem}_{kind}.txt")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.write(f"image={img_path}\n")
                        f.write(f"prompt={effective_prompt}\n\n")
                        f.write(text)
                    # Saving the image once under save_dir for convenience
                    out_img = os.path.join(save_dir, f"{stem}.png")
                    if not os.path.exists(out_img):
                        image.save(out_img)
                except Exception as e:
                    print(c(f"[SAVE-ERROR] Could not save results for {kind}: {e}", FG_RED))

    # Sequential mode completion notice
    print(c("\n[ALL DONE] Finished generating for: ", FG_GREEN) + os.path.basename(img_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
