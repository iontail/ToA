import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class BaseAgent:
    def __init__(self, id: Optional[int] = None, model: Optional[str] = None,
                 max_new_tokens: int = 2048):
        self.id = id
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.reset()

    def reset(self):
        self.claim = ""
        self.options = ""
        self.fact = ""
        self.conclusion = ""
        self.final_decision = ""

    def generate_response(self, messages):
        raise NotImplementedError


class LocalLlama(BaseAgent):
    _model = None
    _tokenizer = None
    _model_path = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = self.model or os.getenv("MODEL_PATH") or os.getenv("TOKENIZER_PATH") or DEFAULT_MODEL
        self.load_model(self.model_path)

    @classmethod
    def load_model(cls, model_path):
        if cls._model is not None and cls._model_path == model_path:
            return

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        cls._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cls._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        cls._model.eval()
        cls._model_path = model_path

        if cls._tokenizer.pad_token_id is None:
            cls._tokenizer.pad_token = cls._tokenizer.eos_token

    def generate_response(self, messages):
        tokenizer = self.__class__._tokenizer
        model = self.__class__._model
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        return tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
