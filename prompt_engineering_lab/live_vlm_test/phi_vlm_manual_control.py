#!/usr/bin/env python3

"""
Manual control for Gym-Duckietown with VLM inference (Phi-3.5-Vision) on ENTER key.

This script mirrors the behavior of vlm_manual_control.py but uses a different
open-source vision-language model: microsoft/Phi-3.5-vision-instruct.

When you press ENTER, the current camera frame is sent to the VLM together with
an instruction prompt. The model returns a description and suggested action.

Dependencies (install in your virtualenv):
  pip install transformers==4.46.3 torch Pillow

Note: Running VLM on CPU is supported but generation may be slow.
"""

from PIL import Image
import os
import logging
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
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
try:
    from transformers.utils import logging as hf_logging  # type: ignore
except Exception:  # pragma: no cover
    hf_logging = None  # type: ignore

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


class VLMRunnerPhi:
    """Wraps the Phi-3.5-Vision-Instruct model for inference on PIL Images."""

    def __init__(self, device: torch.device):
        model_id = "microsoft/Phi-3.5-vision-instruct"

        # Reduce noisy warnings from Transformers and custom transformers_modules
        try:
            if hf_logging is not None:
                hf_logging.set_verbosity_error()
        except Exception:
            pass
        try:
            logging.getLogger("transformers_modules").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
        except Exception:
            pass

        # Load processor with remote code trusted to avoid interactive prompt
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Prepare config and force-disable FlashAttention related flags
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        try:
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = "eager"
            # Common flags used by some custom model implementations
            for name in (
                "use_flash_attention_2",
                "use_flash_attn_2",
                "flash_attn",
                "flash_attention",
            ):
                if hasattr(cfg, name):
                    setattr(cfg, name, False)
        except Exception:
            pass

        # Load model with config and trust remote code; keep float32 and eager attention
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=cfg,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                trust_remote_code=True,
            ).to(device).eval()
        except ImportError as e:
            # If any flash_attn import slipped through, retry once with enforced cfg
            if "flash_attn" in str(e):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=cfg,
                    torch_dtype=torch.float32,
                    attn_implementation="eager",
                    trust_remote_code=True,
                ).to(device).eval()
            else:
                raise
        # Extra safety: update loaded model config/attributes if present
        try:
            if hasattr(self.model.config, "attn_implementation"):
                self.model.config.attn_implementation = "eager"
            for name in (
                "use_flash_attention_2",
                "use_flash_attn_2",
                "flash_attn",
                "flash_attention",
            ):
                if hasattr(self.model.config, name):
                    setattr(self.model.config, name, False)
            if hasattr(self.model, "attn_implementation"):
                self.model.attn_implementation = "eager"
        except Exception:
            pass
        self.device = device

    def _apply_chat_template(self, messages):
        """Apply a chat template if available on the processor or tokenizer.

        This is a best-effort helper for processors that expose chat templates.
        """
        # Prefer processor.apply_chat_template, fall back to tokenizer.apply_chat_template
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "apply_chat_template"):
            return self.processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            # Minimal fallback: naive concatenation
            parts = []
            for m in messages:
                role = m.get("role", "user")
                parts.append(f"{role}:")
                for c in m.get("content", []):
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "image":
                        parts.append("<image>")
            parts.append("assistant:")
            return "\n".join(parts)

    def generate(self, image: Image.Image, user_text: str, max_new_tokens: int = 128) -> str:
        """Run VLM generation for a single image + instruction text and return decoded text.

        For Phi-3.5-Vision, avoid relying on chat_template; build a minimal
        conversation prompt manually using the expected special tokens.
        """
        prompt = (
            "<|user|>\n"
            "<|image_1|>\n"
            f"{user_text}\n"
            "<|end|>\n"
            "<|assistant|>\n"
        )
        # Some processor versions expect singular vs list args; try a few safe variants
        inputs = None
        last_err = None
        for pack in (
            lambda: self.processor(text=prompt, images=image, return_tensors="pt"),
            lambda: self.processor(prompt, image, return_tensors="pt"),
            lambda: self.processor(text=[prompt], images=[image], return_tensors="pt"),
            lambda: self.processor([prompt], [image], return_tensors="pt"),
        ):
            try:
                inputs = pack()
                break
            except Exception as e:
                last_err = e
                continue
        if inputs is None:
            raise RuntimeError(f"Processor packing failed: {repr(last_err)}")

        # Move tensors to device
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        # Track prompt length to decode only newly generated tokens
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
    """Entry point for manual control with Phi VLM integration.

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

    # Prepare VLM (lazy init to display window faster)
    vlm_holder = {"runner": None}
    last_obs = {"img": None}
    vlm_state = {"busy": False}

    def ensure_vlm_loaded():
        if vlm_holder["runner"] is None:
            print("[VLM] Loading microsoft/Phi-3.5-vision-instruct on", device)
            t0 = time.time()
            vlm_holder["runner"] = VLMRunnerPhi(device)
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
                    base = os.path.join(args.vlm_log_dir, f"vlm_phi_{ts}")
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
