import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer

modelID = "./base"

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

quantizationConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(modelID)

tokenizer = AutoTokenizer.from_pretrained(modelID)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        packing=True,
    )

trainer.train()