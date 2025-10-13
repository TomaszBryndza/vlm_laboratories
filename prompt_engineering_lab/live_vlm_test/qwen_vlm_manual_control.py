#!/usr/bin/env python3

"""
Manual control for Gym-Duckietown with VLM inference on ENTER key.

This script mirrors the behavior of test_TinyVLM.py but uses an open-source
vision-language model (Qwen/Qwen2-VL-2B-Instruct) from Hugging Face.

When you press ENTER, the current camera frame is sent to the VLM together with
an instruction prompt. The model returns a description and suggested action.

Dependencies (install in your virtualenv):
  pip install transformers==4.46.3 torch Pillow

Note: Qwen2-VL-2B-Instruct can run on CPU but generation may be slow.
"""

from PIL import Image
import os
from datetime import datetime
import argparse
import sys
import time
import warnings

import gym
import numpy as np
import pyglet
from pyglet.window import key

import torch
from transformers import AutoProcessor

# Qwen2-VL specific import (class is available via transformers>=4.43)
try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception as e:  # pragma: no cover
    Qwen2VLForConditionalGeneration = None

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


class VLMRunnerQwen:
    """Wraps the Qwen2-VL model and processor for easy inference on PIL Images."""

    def __init__(self, device: torch.device):
        if Qwen2VLForConditionalGeneration is None:
            raise RuntimeError(
                "Qwen2VLForConditionalGeneration not available. Please install transformers>=4.43."
            )

        model_id = "Qwen/Qwen2-VL-2B-Instruct"

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        # Force float32 on CPU for broad compatibility
        self.model = (
            Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float32, trust_remote_code=True
            )
            .to(device)
            .eval()
        )
        self.device = device

    def generate(self, image: Image.Image, user_text: str, max_new_tokens: int = 128) -> str:
        """Run VLM generation for a single image + instruction text.

        Args:
            image: PIL Image containing the current frame.
            user_text: Instructional text prompt.
            max_new_tokens: Generation length budget.

        Returns:
            The decoded text output from the model.
        """
        # Preferred chat-style messages API (Transformers >= 4.44 supports this)
        try:
            # Preferred chat-style API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]

            inputs = self.processor(
                messages=messages,
                images=[image],
                return_tensors="pt",
            )
        except Exception:
            # Fallback: compose a prompt with chat template + separate images
            try:
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
                ]
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                inputs = self.processor(
                    text=[text], images=[image], return_tensors="pt"
                )
            except Exception as inner_e:
                raise RuntimeError(f"Processor formatting failed: {inner_e}")

        # Move tensors to device
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Remember prompt length so we can strip it from the decoded text
        prompt_len = None
        try:
            if isinstance(inputs.get("input_ids", None), torch.Tensor):
                prompt_len = int(inputs["input_ids"].shape[1])
        except Exception:
            prompt_len = None

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        # If the model returned prompt+completion, drop the prompt part
        try:
            if prompt_len is not None and output_ids.shape[1] > prompt_len:
                gen_ids = output_ids[:, prompt_len:]
            else:
                gen_ids = output_ids
        except Exception:
            gen_ids = output_ids

        out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return out_text


def main():
    """Entry point for manual control with Qwen2-VL VLM integration.

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
                        help="How often to print step_count (0 = never, matches manual_control silence during VLM runs)")
    parser.add_argument("--vlm-log-dir", default="vlm_logs", type=str,
                        help="Directory to write full VLM outputs and the used frame")
    args = parser.parse_args()

    # Prefer CPU for portability; use CUDA if explicitly desired and available
    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    # Suppress spurious warnings from backends
    warnings.filterwarnings("ignore", category=UserWarning)

    env = build_env(args)
    env.reset()
    env.render()
    print("Controls: arrows to drive, LSHIFT to boost, BACKSPACE to reset, ESC to exit. Press ENTER to run VLM on the current frame.")

    # Prepare VLM (lazily instantiate to show window sooner)
    vlm_holder = {"runner": None}
    # Keep last observation to run VLM from on_key_press without blocking the update loop
    last_obs = {"img": None}
    # Track VLM activity to avoid interleaving noisy prints
    vlm_state = {"busy": False}

    def ensure_vlm_loaded():
        if vlm_holder["runner"] is None:
            print("[VLM] Loading Qwen2-VL-2B-Instruct on", device)
            t0 = time.time()
            vlm_holder["runner"] = VLMRunnerQwen(device)
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
            # Run VLM once on the latest observation (use ENTER to avoid interfering with movement)
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

                # Persist full outputs and associated frame to a log file
                try:
                    os.makedirs(args.vlm_log_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base = os.path.join(args.vlm_log_dir, f"vlm_{ts}")
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
            except Exception as e:  # pragma: no cover
                print("[VLM ERROR]", repr(e))
            finally:
                vlm_state["busy"] = False

    # Register a keyboard handler (after on_key_press like in manual_control)
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
        # Do not force-stop on SPACE; we use it to trigger VLM in on_key_press

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

        # Store last observation for SPACE-triggered VLM
        last_obs["img"] = obs

        # Optional step logging; avoid spamming during VLM runs
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
