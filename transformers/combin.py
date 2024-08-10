from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./base"
lora_name = "./lora/checkpoint-30"
target = "merged"

# Load the base model
# base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda")

# Load the LoRA model
# peft_model = PeftModelForCausalLM.from_pretrained(base_model, "./lora/checkpoint-30", device_map="cuda", torch_dtype=torch.bfloat16)
peft_model = PeftModelForCausalLM.from_pretrained(base_model, lora_name, device_map="cuda")

# Merge LoRA weights with base model
model = peft_model.merge_and_unload()
# model = peft_model.base_model.model

# Save the merged model
model.save_pretrained(target)

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(target)
