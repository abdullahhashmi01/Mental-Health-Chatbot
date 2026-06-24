"""
src/pipeline/prediction_pipeline.py
---------------------------------------
Loads the fine-tuned model once and serves replies. Used by both
chat_cli.py and app_streamlit.py so there's a single source of
inference logic (no duplicated generation code).
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.logger import logging
from src.exception import ChatbotException
from src.entity.config_entity import PredictionConfig
from src.components.safety_filter import SafetyFilter


class PredictionPipeline:
    def __init__(self, config: PredictionConfig = PredictionConfig()):
        try:
            self.config = config
            self.safety_filter = SafetyFilter()

            logging.info(f"Loading fine-tuned model from: {config.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(config.model_path)
            self.model.eval()
        except Exception as e:
            raise ChatbotException(e, sys)

    def generate_reply(self, user_message: str, max_new_tokens: int = 60) -> str:
        try:
            # Safety check always takes priority over free-form generation
            if self.safety_filter.is_crisis_message(user_message):
                logging.info("Crisis language detected -> returning fixed safety response")
                return self.safety_filter.get_crisis_response()

            prompt = (
                f"{self.config.system_prompt}\n\n"
                f"User: {user_message}\n"
                f"Assistant:"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    temperature=0.8,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            reply = full_text.split("Assistant:")[-1].strip()
            reply = reply.split("User:")[0].strip()

            if not reply:
                reply = "I'm here for you. Could you tell me a little more about how you're feeling?"

            return reply
        except Exception as e:
            raise ChatbotException(e, sys)
