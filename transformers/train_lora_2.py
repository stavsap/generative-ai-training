# https://github.com/rohan-paul/LLM-FineTuning-Large-Language-Models/blob/main/Mistral_7B_Instruct_GPTQ_finetune.ipynb

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import GPTQConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

dataset = load_dataset('gbharti/finance-alpaca')
# Split the dataset into train and test sets
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Further split the train dataset into train and validation sets
train_val_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']



##############

pretrained_model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"


def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def format_input_data_to_build_model_prompt(data_point):
        instruction = str(data_point['instruction'])
        input_query = str(data_point['input'])
        response = str(data_point['output'])

        if len(input_query.strip()) == 0:
            full_prompt_for_model = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction} \n\n### Input:\n{input_query}\n\n### Response:\n{response}"""

        else:
            full_prompt_for_model = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"""
        return tokenize(full_prompt_for_model)


def build_qlora_model(
    pretrained_model_name_or_path: str = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ",
    gradient_checkpointing: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
    """
    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
        gradient_checkpointing (bool): Whether to use gradient checkpointing or not.
        cache_dir (Optional[Path]): The directory to cache the model in.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the built model and tokenizer.
    """

    # If I am using any GPTQ model, then need to comment-out bnb_config
    # as I can not quantize an already quantized model

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # In below as well, when using any GPTQ model
    # comment-out the quantization_config param

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config_loading = GPTQConfig(bits=4, use_exllama=False, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        # quantization_config=bnb_config,
        quantization_config=quantization_config_loading,
        device_map="auto",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    #disable tensor parallelism
    model.config.pretraining_tp = 1

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = (
            False  # Gradient checkpointing is not compatible with caching.
        )
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True  # It is good practice to enable caching when using the model for inference.

    return model, tokenizer