#!/usr/bin/env python3
"""Vision-Language Model runner implementations for Duckietown manual control.

Each runner exposes a unified generate(image: PIL.Image, user_text: str, max_new_tokens: int = 128) -> str
interface so the simulator can remain model-agnostic.

Available model keys for factory:
  phi   -> microsoft/Phi-3.5-vision-instruct
  qwen  -> Qwen/Qwen2-VL-2B-Instruct
  tiny  -> anananan116/TinyVLM
"""
from __future__ import annotations

from typing import Dict, Type, Optional
from PIL import Image
import os
import re
import time
import logging
import warnings
import json

import torch

# Transformers imports (lazy error handling for optional models)
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig, AutoTokenizer,AutoModelForVision2Seq
try:  # Qwen specific
    from transformers import Qwen2VLForConditionalGeneration  # type: ignore
except Exception:  # pragma: no cover
    Qwen2VLForConditionalGeneration = None  # type: ignore

# Reduce noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
try:
    logging.getLogger("transformers_modules").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
except Exception:
    pass

# ------------------------- Phi Runner ------------------------- #
class VLMRunnerPhi:
    """Wrapper for microsoft/Phi-3.5-vision-instruct."""
    def __init__(self, device: torch.device):
        model_id = "microsoft/Phi-3.5-vision-instruct"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Force eager / disable flash attention flags for portability
        for name in ("attn_implementation",):
            if hasattr(cfg, name):
                setattr(cfg, name, "eager")
        for name in ("use_flash_attention_2", "use_flash_attn_2", "flash_attn", "flash_attention"):
            if hasattr(cfg, name):
                setattr(cfg, name, False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=cfg,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(device).eval()
        self.device = device

    def generate(
        self,
        image: Image.Image,
        user_text: str,
        max_new_tokens: int = 128,
        force_json: bool = False,
        enforce_rule_id: Optional[str] = None,
    ) -> str:
        """Generate text from Phi VLM.

        Enhancements over original implementation:
        - Robust structured-mode detection & balanced JSON extraction (prevents trailing hallucinated prose like 'Article: ...').
        - Optional rule_id enforcement (overwrite or flag if model outputs a different ID).
        - Optionally force JSON parsing even if trigger phrase is absent.
        """
        structured_trigger_phrases = [
            'return json only', 'strict json', 'json only', 'strictly json','return json'
        ]
        lower_user = user_text.lower()
        is_structured = any(p in lower_user for p in structured_trigger_phrases) or force_json

        # Minor prompt hardening: explicitly restate single-object expectation when structured
        hardener = "\nRespond with ONLY one JSON object and nothing after it." if is_structured else ""
        prompt = (
            "<|user|>\n"
            "<|image_1|>\n"
            f"{user_text}{hardener}\n"
            "<|end|>\n"
            "<|assistant|>\n"
        )

        inputs = None
        last_err = None
        for pack in (
            lambda: self.processor(text=prompt, images=image, return_tensors="pt"),
            lambda: self.processor(prompt, image, return_tensors="pt"),
            lambda: self.processor(text=[prompt], images=[image], return_tensors="pt"),
            lambda: self.processor([prompt], [image], return_tensors="pt"),
        ):
            try:
                inputs = pack(); break
            except Exception as e:  # try fallbacks
                last_err = e
                continue
        if inputs is None:
            raise RuntimeError(f"Processor packing failed: {repr(last_err)}")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        prompt_len = None
        if isinstance(inputs.get("input_ids"), torch.Tensor):
            prompt_len = int(inputs["input_ids"].shape[1])

        gen_kwargs = {
            'max_new_tokens': min(max_new_tokens, 56) if is_structured else max_new_tokens,
            'do_sample': not is_structured,
        }
        if is_structured:
            gen_kwargs.update({'temperature': 0.0, 'top_p': 1.0})

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        if prompt_len is not None and output_ids.shape[1] > prompt_len:
            gen_ids = output_ids[:, prompt_len:]
        else:
            gen_ids = output_ids
        raw_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        if not is_structured:
            return raw_text

        # --- Robust JSON object extraction --- #
        def _balanced_first_object(s: str) -> Optional[str]:
            start = s.find('{')
            if start == -1:
                return None
            depth = 0
            for i, ch in enumerate(s[start:], start=start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i+1]
            return None

        extracted = _balanced_first_object(raw_text) or raw_text
        trimmed = extracted.strip()

        # Remove any trailing characters after the balanced object (already isolated, but be safe)
        m = re.match(r'\s*(\{.*\})', trimmed, re.DOTALL)
        if m:
            trimmed = m.group(1)

        # Attempt to parse & clean JSON: restrict keys to rule_id, score (others dropped)
        cleaned_output = trimmed
        try:
            obj = json.loads(trimmed)
            if isinstance(obj, dict):
                # Normalize rule_id
                if enforce_rule_id is not None:
                    obj['rule_id'] = enforce_rule_id
                # Clamp score to allowed discrete set if present
                allowed_scores = {0,10,20,30,40,50,60,70,80,90,100}
                if 'score' in obj:
                    try:
                        # Round to nearest allowed bucket
                        val = int(obj['score'])
                        if val not in allowed_scores:
                            # snap to nearest
                            val = min(allowed_scores, key=lambda x: abs(x - val))
                        obj['score'] = val
                    except Exception:
                        obj['score'] = 0
                # Reconstruct minimal JSON
                minimal = {k: obj[k] for k in ('rule_id','score') if k in obj}
                cleaned_output = json.dumps(minimal, ensure_ascii=False)
        except Exception:
            # Fall back: ensure we at least cut after first closing brace
            brace_idx = trimmed.find('}')
            if brace_idx != -1:
                cleaned_output = trimmed[:brace_idx+1]

        return cleaned_output

# ------------------------- Qwen Runner ------------------------- #
class VLMRunnerQwen25:
    """Wrapper for Qwen/Qwen2.5-VL-* (e.g., 3B/7B/32B) Instruct checkpoints."""
    def __init__(
        self,
        device: torch.device,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            device: torch.device, e.g. torch.device("cuda") or torch.device("cpu")
            model_id: HF repo id for Qwen2.5-VL Instruct (e.g. "Qwen/Qwen2.5-VL-7B-Instruct")
            dtype: optional torch dtype; if None, picks float16 on CUDA, else float32
        """
        self.device = device
        if dtype is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Processor i model (Qwen2.5-VL używa AutoModelForVision2Seq + trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = (
            AutoModelForVision2Seq
            .from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
            .to(device)
            .eval()
        )

    def _build_messages(self, user_text: str) -> List[Dict[str, Any]]:
        # Ten sam format wiadomości co w Qwen2-VL: obraz + tekst
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

    def generate(
        self,
        image: Image.Image,
        user_text: str,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Zwraca wygenerowany tekst odpowiedzi dla (obraz, prompt).

        Uwaga: Qwen2.5-VL akceptuje zarówno `messages=...` jak i surowy text z apply_chat_template.
        """
        messages = self._build_messages(user_text)

        # Spróbuj ścieżki z messages=..., a jeśli procesor wymaga template — użyj apply_chat_template
        try:
            inputs = self.processor(
                messages=messages,
                images=[image],
                return_tensors="pt"
            )
        except Exception:
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            )

        # Przenieś tensory na docelowe urządzenie
        inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        # Długość promptu (jeśli dostępna) — żeby odciąć prefiks wejściowy
        prompt_len = None
        if isinstance(inputs.get("input_ids"), torch.Tensor):
            prompt_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        # Wytnij same tokeny generacji (jeśli znamy długość promptu)
        if prompt_len is not None and output_ids.shape[1] > prompt_len:
            gen_ids = output_ids[:, prompt_len:]
        else:
            gen_ids = output_ids

        out_text = self.processor.batch_decode(
            gen_ids, skip_special_tokens=True
        )[0].strip()

        return out_text
    
class VLMRunnerQwen:
    """Wrapper for Qwen/Qwen2-VL-2B-Instruct."""
    def __init__(self, device: torch.device):
        if Qwen2VLForConditionalGeneration is None:
            raise RuntimeError("Qwen2VLForConditionalGeneration unavailable; install transformers>=4.43")
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
        # ).to(device).eval()
        self.device = device

    def generate(self, image: Image.Image, user_text: str, max_new_tokens: int = 128) -> str:
        try:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
            inputs = self.processor(messages=messages, images=[image], return_tensors="pt")
        except Exception:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        prompt_len = None
        if isinstance(inputs.get("input_ids"), torch.Tensor):
            prompt_len = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
        if prompt_len is not None and output_ids.shape[1] > prompt_len:
            gen_ids = output_ids[:, prompt_len:]
        else:
            gen_ids = output_ids
        out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return out_text

# ------------------------- TinyVLM Runner ------------------------- #
class VLMRunnerTiny:
    """Wrapper for anananan116/TinyVLM."""
    def __init__(self, device: torch.device):
        model_id = "anananan116/TinyVLM"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        ).to(device).eval()
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.model.resize_token_embeddings(len(self.tokenizer))
        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.model.config, "eos_token_id", None) is None and self.tokenizer.eos_token_id is not None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self._pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        self._eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
        self.device = device

    def generate(self, image: Image.Image, user_text: str, max_new_tokens: int = 128) -> str:
        prompt = "The image provided has been provided:<IMGPLH>. " + user_text
        inputs = self.model.prepare_input_ids_for_generation([prompt], [image], self.tokenizer)
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
        gen_seq = output_ids[0]
        input_len = int(inputs["input_ids"].shape[-1])
        new_tokens = gen_seq[input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        m = re.search(r"(?is)(?:^|\n)assistant\s*\n", text)
        if m:
            text = text[m.end():].strip()
        for token in ("<IMAGE>", "<IMAGE_END>", "<IMAGE><IMAGE_END>"):
            text = text.replace(token, "")
        lines = text.splitlines()
        changed = True
        while lines and changed:
            changed = False
            if lines[0].strip().lower() in ("system", "user"):
                lines.pop(0); changed = True
                while lines and lines[0].strip() != "":
                    lines.pop(0)
                if lines and lines[0].strip() == "":
                    lines.pop(0)
        text = "\n".join(lines).strip()
        return text

# ------------------------- Factory ------------------------- #
RUNNER_MAP: Dict[str, Type] = {
    "phi": VLMRunnerPhi,
    "qwen": VLMRunnerQwen,
    "tiny": VLMRunnerTiny,
}

def get_vlm_runner(name: str, device: torch.device):
    key = name.lower().strip()
    if key not in RUNNER_MAP:
        raise ValueError(f"Unknown VLM model '{name}'. Available: {sorted(RUNNER_MAP.keys())}")
    t0 = time.time()
    print(f"[VLM] Loading model '{key}' on {device} ...")
    runner = RUNNER_MAP[key](device)
    print(f"[VLM] Model '{key}' loaded in {time.time() - t0:.1f}s")
    return runner

__all__ = [
    "VLMRunnerPhi", "VLMRunnerQwen", "VLMRunnerTiny", "get_vlm_runner"
]
