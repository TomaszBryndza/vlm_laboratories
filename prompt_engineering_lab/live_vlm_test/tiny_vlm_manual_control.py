#!/usr/bin/env python3

"""
Manual control for Gym-Duckietown with VLM inference (TinyVLM) on ENTER key.

This script mirrors the structure of the other VLM manual control tools.
The VLM functionality is wrapped in a class and triggered by pressing ENTER.

Dependencies (install in your virtualenv):
  pip install transformers==4.46.3 torch Pillow

Note: Running VLM on CPU is supported but generation may be slow.
"""

from PIL import Image
import os
from datetime import datetime
import argparse
import sys
import time
import warnings
import re

import gym
import numpy as np
import pyglet
from pyglet.window import key

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gym_duckietown.envs import DuckietownEnv


# Prompts used for the VLM
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
    """Create a Duckietown or Gym environment from CLI arguments.

    Args:
        args: Parsed command-line arguments containing environment options.

    Returns:
        The initialized Gym environment instance.
    """
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


class VLMRunnerTiny:
    """Wraps the TinyVLM model (anananan116/TinyVLM) for inference on PIL Images."""

    def __init__(self, device: torch.device):
        model_id = "anananan116/TinyVLM"
        # Load tokenizer and model; trust_remote_code to enable custom helper methods
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(device).eval()
        self.device = device

        # Ensure pad/eos tokens are defined to avoid generation warnings
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                # Resize model embeddings if new token added
                self.model.resize_token_embeddings(len(self.tokenizer))
        # Propagate to model config
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.model.config, "eos_token_id", None) is None and self.tokenizer.eos_token_id is not None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        # Cache ids
        self._pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        self._eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None

    def generate(self, image: Image.Image, user_text: str, max_new_tokens: int = 128) -> str:
        """Run VLM generation for a single image + instruction text and return decoded text."""
        # TinyVLM expects a single <IMGPLH> token in the prompt for the image
        prompt = (
            "The image provided has been provided:<IMGPLH>. "
            f"{user_text}"
        )

        # Model provides a convenience utility for packing text+image
        inputs = self.model.prepare_input_ids_for_generation([prompt], [image], self.tokenizer)
        # Move tensors to device
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                encoded_image=inputs["encoded_image"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=self._pad_id,
                eos_token_id=self._eos_id,
            )

        # Decode only the newly generated tokens (exclude the prompt part)
        gen_seq = output_ids[0]
        input_len = int(inputs["input_ids"].shape[-1])
        new_tokens = gen_seq[input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean potential chat-style preamble if produced by the model
        # e.g., leading "system...", "user...", and keep only after the "assistant" header
        m = re.search(r"(?is)(?:^|\n)assistant\s*\n", text)
        if m:
            text = text[m.end():].strip()

        # Remove any leftover image placeholders
        for token in ("<IMAGE>", "<IMAGE_END>", "<IMAGE><IMAGE_END>"):
            text = text.replace(token, "")

        # Drop initial role/meta lines like 'system' or 'user' blocks if present
        lines = text.splitlines()
        changed = True
        while lines and changed:
            changed = False
            if lines[0].strip().lower() in ("system", "user"):
                # remove role label line and subsequent meta lines until blank line
                lines.pop(0)
                changed = True
                while lines and lines[0].strip() != "":
                    lines.pop(0)
                if lines and lines[0].strip() == "":
                    lines.pop(0)
        text = "\n".join(lines).strip()

        return text


def main():
    """Entry point for manual control with TinyVLM integration.

    Initializes the environment and registers keyboard handlers. Press ENTER
    to run the VLM on the latest camera frame and print/save its outputs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--map-name", default="udem1")
    parser.add_argument("--distortion", default=False, action="store_true")
    parser.add_argument("--camera_rand", default=False, action="store_true")
    parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
    parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
    parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
    parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
    parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--max-new-tokens", default=128, type=int, help="VLM generation length")
    parser.add_argument("--print-steps-every", default=0, type=int,
                        help="How often to print step_count (0 = never)")
    parser.add_argument("--vlm-log-dir", default="vlm_logs", type=str,
                        help="Directory to write full VLM outputs and the used frame")
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    warnings.filterwarnings("ignore", category=UserWarning)

    env = build_env(args)
    env.reset()
    env.render()
    print("Controls: arrows to drive, LSHIFT to boost, BACKSPACE to reset, ESC to exit. Press ENTER to run VLM on the current frame.")

    # Lazy model init for faster window display
    vlm_holder = {"runner": None}
    last_obs = {"img": None}
    vlm_state = {"busy": False}

    def ensure_vlm_loaded():
        if vlm_holder["runner"] is None:
            print("[VLM] Loading anananan116/TinyVLM on", device)
            t0 = time.time()
            vlm_holder["runner"] = VLMRunnerTiny(device)
            print(f"[VLM] Loaded in {time.time() - t0:.1f}s")

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print("RESET")
            env.reset()
            env.render()
        elif symbol == key.PAGEUP:
            env.unwrapped.cam_angle[0] = 0
        elif symbol == key.ESCAPE:
            env.close()
            sys.exit(0)
        elif symbol == key.RETURN:
            try:
                vlm_state["busy"] = True
                ensure_vlm_loaded()
                if last_obs["img"] is None:
                    print("[VLM] No frame yet; please wait a moment and press ENTER again.")
                    return
                image = Image.fromarray(last_obs["img"]).convert("RGB")
                print("\n==================== VLM START ====================", flush=True)
                print("[ACTION] ENTER pressed -> running VLM", flush=True)
                desc_text = vlm_holder["runner"].generate(
                    image, PROMPT_DESCRIPTION, max_new_tokens=args.max_new_tokens
                )
                print("\n[VLM Description]\n" + desc_text, flush=True)
                decision_text = vlm_holder["runner"].generate(
                    image, PROMPT_DECISION, max_new_tokens=args.max_new_tokens
                )
                print("\n[VLM Decision]\n" + decision_text, flush=True)

                # Persist outputs
                try:
                    os.makedirs(args.vlm_log_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base = os.path.join(args.vlm_log_dir, f"vlm_tiny_{ts}")
                    img_path = base + ".png"
                    txt_path = base + ".txt"
                    image.save(img_path)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(f"env={args.env_name} map={args.map_name}\n")
                        f.write("\n[VLM Description]\n")
                        f.write(desc_text)
                        f.write("\n\n[VLM Decision]\n")
                        f.write(decision_text)
                    print(f"\n[VLM] Saved frame -> {img_path}\n[VLM] Saved text  -> {txt_path}", flush=True)
                except Exception as log_e:
                    print("[VLM] Failed to write logs:", repr(log_e))

                print("===================== VLM END =====================\n", flush=True)
            except Exception as e:
                print("[VLM ERROR]", repr(e))
            finally:
                vlm_state["busy"] = False

    # Keyboard handler for driving
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)

    def update(dt):
        wheel_distance = 0.102
        min_rad = 0.08
        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action += np.array([0.44, 0.0])
        if key_handler[key.DOWN]:
            action -= np.array([0.44, 0])
        if key_handler[key.LEFT]:
            action += np.array([0, 1])
        if key_handler[key.RIGHT]:
            action -= np.array([0, 1])

        v1 = action[0]
        v2 = action[1]
        # Limit radius of curvature
        if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
            delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
            v1 += delta_v
            v2 -= delta_v

        action[0] = v1
        action[1] = v2

        if key_handler[key.LSHIFT]:
            action *= 1.5

        obs, reward, done, info = env.step(action)
        last_obs["img"] = obs

        # Optional step logging
        if args.print_steps_every and not vlm_state["busy"]:
            step = getattr(env.unwrapped, "step_count", None)
            if step is not None and step % max(1, args.print_steps_every) == 0:
                try:
                    print("step_count = %s, reward=%.3f" % (step, reward))
                except Exception:
                    pass

        if done:
            print("done!")
            env.reset()
            env.render()

        env.render()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
    pyglet.app.run()
    env.close()


if __name__ == "__main__":
    main()
