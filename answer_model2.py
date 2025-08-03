#
import os
import re
import time
import torch
from typing import Optional, Union, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from cerebras.cloud.sdk import Cerebras

torch.random.manual_seed(0)
from cerebras.cloud.sdk import Cerebras

class AAgent:
    def __init__(self, adapter_type: str, api_key: str):
        self.adapter_type = adapter_type
        self.cerebras_api_key = api_key
        self.cerebras_client = Cerebras(api_key=self.cerebras_api_key)
 
        #Local installation
        except ImportError:
            self.cerebras_client = None
            print("Cerebras SDK not installed. Cerebras features will be unavailable.")
            # self.model_type = input("Available models: Qwen3-1.7B and Qwen3-4B. Please enter 1.7B or 4B: ").strip()
            self.model_type = kwargs.get('model_type', '4B').strip()
            # model_name = "Qwen/Qwen3-4B"
            model_name = "/jupyter-tutorial/hf_models/Qwen3-4B"
            
            # load the tokenizer and the model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )

            self.adapter_type = adapter_type
            
            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = self._setup_model_with_adapter(base_model)        
            self.model = self.model.eval()

    def _setup_model_with_adapter(self, base_model):
        """Setup model with or without adapter based on configuration."""
        if self.adapter_type is None:
            print("No adapter specified - using base model")
            return base_model

        self.adapter_type = self.adapter_type.lower()

        if self.adapter_type == 'sft' or self.adapter_type == 'grpo':
            print("Loading required adapter")
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                print(f"No trained checkpoints found for {self.adapter_type}")
                inference_model = base_model
            else:
                print(f"Loading LoRA adapters from: {checkpoint_path}")
                inference_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            return inference_model
        else:
            print(f"Unknown adapter type: {self.adapter_type}. Using base model.")
            return base_model

    def cerebras_chat_completion(self, messages, model="llama-4-scout-17b-16e-instruct"):
        """Use Cerebras API for chat completion. Returns the response content or error message."""
        if self.cerebras_client is None:
            return "Cerebras client not available. Please install the SDK and set the API key."
        try:
            result = self.cerebras_client.chat.completions.create(
                messages=messages,
                model=model,
            )
            # Fix: access message content as attribute, not as dict
            if hasattr(result, "choices") and result.choices:
                return result.choices[0].message.content
            return str(result)
        except Exception as e:
            return f"Cerebras API error: {e}"

   
        
if __name__ == "__main__":
    """
    Usage:
    How to run as main file?
        a. python -m agents.answer_model2
        b. change the adapter_type as shown: `AAgent(adapter_type=None) # adapter_type can be "sft" or "grpo"`
    """

    # Single message (backward compatible)
    ans_agent = AAgent(adapter_type=None) # adapter_type can be "sft" or "grpo"
    response = ans_agent.cerebras_chat_completion(([
            {"role": "user", "content": "Write a Triton kernel for matrix multiplication that can handle 1024x1024 matrices efficiently."}
        ]))
    print(f"Single response: {response}")
    print("-----------------------------------------------------------")
          
    # Batch processing (new capability)
    messages = [
        "Implement a Triton kernel for vector addition with proper memory coalescing.",
        "Write a Triton kernel that performs element-wise multiplication of two tensors.",
        "Create a Triton kernel for softmax operation that can work on GPU.",
        "Develop a Triton kernel for convolution operations with configurable filter sizes.",
        "Write a Triton kernel that implements a fused attention mechanism.",
    ]
    responses = ans_agent.cerebras_chat_completion(messages)
    print("Responses:")
    #for i, resp in enumerate(responses):
        #print(f"Message {i+1}: {resp}")
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.cerebras_chat_completion(
        [{"role": "user", "content": "Create a Triton kernel for efficient matrix multiplication with block size optimization."}]
    )
    print(f"Custom response: {response}")
