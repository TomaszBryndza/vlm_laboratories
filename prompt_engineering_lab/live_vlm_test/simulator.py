#!/usr/bin/env python3
"""Generic Duckietown manual control simulator with pluggable VLM models.

Use --vlm-model to select which model implementation to use (phi, qwen, tiny).
The actual runner classes live in vlm_runners.py; this module focuses on:
  * Environment creation & keyboard driving loop
  * Capturing frames and invoking VLM on ENTER
  * Logging outputs & frame images

Usage examples:
  python simulator.py --vlm-model phi
  python simulator.py --vlm-model qwen --map-name udem1

Legacy wrappers (phi_vlm_manual_control.py etc.) now delegate here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
import warnings

from typing import Optional

import numpy as np
import gym
import pyglet
from pyglet.window import key
from PIL import Image
import torch

from gym_duckietown.envs import DuckietownEnv
from vlm_runners import get_vlm_runner

PROMPT_DECISION = (
    "What action should be taken by the vehicle in this situation? "
    "Provide a decision based on the current state of the environment. "
    "Describe what you see, the action to take, and why."
)
PROMPT_DESCRIPTION = (
    "Describe the image in detail. Focus on the environment, road, obstacles, vehicles, "
    "pedestrians, and any relevant details that help understand the scene."
)


def build_env(args: argparse.Namespace):
    if args.env_name and args.env_name.find("Duckietown") != -1:
        env = DuckietownEnv(
            seed=args.seed,
            map_name=args.map_name,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
            camera_rand=args.camera_rand,
            dynamics_rand=args.dynamics_rand,
        )
    else:
        env = gym.make(args.env_name)
    return env


def run_manual_control(vlm_model: str, args: Optional[argparse.Namespace] = None):
    """Run the manual control simulator with the chosen VLM model.

    Parameters
    ----------
    vlm_model : str
        One of: 'phi', 'qwen', 'tiny'.
    args : argparse.Namespace, optional
        Pre-parsed arguments; if None new CLI parsing will occur.
    """
    if args is None:
        parser = build_arg_parser()
        # If caller provided vlm_model certainty and not in argv, inject default
        if '--vlm-model' not in ' '.join(sys.argv[1:]):
            sys.argv.append(f'--vlm-model={vlm_model}')
        args = parser.parse_args()
    else:
        # override with explicit model if provided separately
        args.vlm_model = vlm_model

    # Device selection (keep CPU for broad portability; could extend later)
    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    warnings.filterwarnings("ignore", category=UserWarning)

    env = build_env(args)
    env.reset(); env.render()
    print("Controls: arrows to drive, LSHIFT to boost, BACKSPACE to reset, ESC to exit. Press ENTER for VLM.")

    # Lazy VLM model load
    vlm_holder = {"runner": None}
    last_obs = {"img": None}
    vlm_state = {"busy": False}

    def ensure_vlm_loaded():
        if vlm_holder["runner"] is None:
            vlm_holder["runner"] = get_vlm_runner(args.vlm_model, device)

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print("RESET"); env.reset(); env.render(); return
        if symbol == key.PAGEUP:
            env.unwrapped.cam_angle[0] = 0; return
        if symbol == key.ESCAPE:
            env.close(); sys.exit(0)
        if symbol == key.RETURN:
            try:
                vlm_state["busy"] = True
                ensure_vlm_loaded()
                if last_obs["img"] is None:
                    print("[VLM] No frame yet; wait then press ENTER again."); return
                image = Image.fromarray(last_obs["img"]).convert("RGB")
                print("\n==================== VLM START ====================", flush=True)
                desc_text = vlm_holder["runner"].generate(image, PROMPT_DESCRIPTION, max_new_tokens=args.max_new_tokens)
                print("\n[VLM Description]\n" + desc_text, flush=True)
                decision_text = vlm_holder["runner"].generate(image, PROMPT_DECISION, max_new_tokens=args.max_new_tokens)
                print("\n[VLM Decision]\n" + decision_text, flush=True)
                # Logging
                try:
                    os.makedirs(args.vlm_log_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base = os.path.join(args.vlm_log_dir, f"vlm_{args.vlm_model}_{ts}")
                    image.save(base + '.png')
                    with open(base + '.txt', 'w', encoding='utf-8') as f:
                        f.write(f"env={args.env_name} map={args.map_name} model={args.vlm_model}\n")
                        f.write("\n[VLM Description]\n" + desc_text)
                        f.write("\n\n[VLM Decision]\n" + decision_text)
                    print(f"\n[VLM] Saved frame -> {base + '.png'}\n[VLM] Saved text  -> {base + '.txt'}", flush=True)
                except Exception as e:
                    print("[VLM] Log write failed:", repr(e))
                print("===================== VLM END =====================\n", flush=True)
            except Exception as e:  # pragma: no cover
                print("[VLM ERROR]", repr(e))
            finally:
                vlm_state["busy"] = False

    key_handler = key.KeyStateHandler(); env.unwrapped.window.push_handlers(key_handler)

    def update(dt):
        wheel_distance = 0.102
        min_rad = 0.08
        action = np.array([0.0, 0.0])
        if key_handler[key.UP]: action += np.array([0.44, 0.0])
        if key_handler[key.DOWN]: action -= np.array([0.44, 0])
        if key_handler[key.LEFT]: action += np.array([0, 1])
        if key_handler[key.RIGHT]: action -= np.array([0, 1])
        v1, v2 = action
        if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
            delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
            v1 += delta_v; v2 -= delta_v
        action[0], action[1] = v1, v2
        if key_handler[key.LSHIFT]: action *= 1.5
        obs, reward, done, info = env.step(action)
        last_obs["img"] = obs
        if args.print_steps_every and not vlm_state["busy"]:
            step = getattr(env.unwrapped, "step_count", None)
            if step is not None and step % max(1, args.print_steps_every) == 0:
                try:
                    print(f"step_count = {step}, reward={reward:.3f}")
                except Exception:
                    pass
        if done:
            print("done!"); env.reset(); env.render()
        env.render()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
    pyglet.app.run(); env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--env-name", default="Duckietown-v0")
    p.add_argument("--map-name", default="udem1")
    p.add_argument("--distortion", default=False, action="store_true")
    p.add_argument("--camera_rand", default=False, action="store_true")
    p.add_argument("--draw-curve", action="store_true")
    p.add_argument("--draw-bbox", action="store_true")
    p.add_argument("--domain-rand", action="store_true")
    p.add_argument("--dynamics_rand", action="store_true")
    p.add_argument("--frame-skip", default=1, type=int)
    p.add_argument("--seed", default=1, type=int)
    p.add_argument("--max-new-tokens", default=128, type=int)
    p.add_argument("--print-steps-every", default=0, type=int)
    p.add_argument("--vlm-log-dir", default="vlm_logs", type=str)
    p.add_argument("--vlm-model", default="phi", choices=["phi", "qwen", "tiny"], help="Select VLM model")
    return p


def main():  # CLI entry
    parser = build_arg_parser(); args = parser.parse_args()
    run_manual_control(args.vlm_model, args)

if __name__ == "__main__":
    main()
