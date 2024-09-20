import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from dotenv import load_dotenv
import os
load_dotenv()

model_path =  os.getenv('BASE_MODEL')
data_path = "./data"
target_lora_path = "./lora"
qlora = False
quantize_config = None

if qlora:
    quantize_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

model.train()  # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)

print(model)

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","o_proj","v_proj","down_proj","gate_proj","up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

config.inference_mode = False

# LoRA trainable version of model
model = get_peft_model(model, config)

from auto_gptq import exllama_set_max_input_length
model = exllama_set_max_input_length(model, max_input_length=1024 * 16)

# trainable parameter count
model.print_trainable_parameters()

data = load_dataset(data_path)

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["example"]

    #tokenize and truncate text
    # tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length"
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)

# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# hyperparameters
lr = 1e-5
batch_size = 32
num_epochs = 3

# define training arguments
training_args = TrainingArguments(
    output_dir= target_lora_path,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.1,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=7,
    fp16=False,
    optim="paged_adamw_8bit",
)

# configure trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)

trainer.train()
# trainer.save_model(target_lora_path)
# tokenizer.save_pretrained(target_lora_path)

model.eval()  # Set the model to evaluation mode

# Prepare input
chat = [
    {"role": "user", "content": "What is Mojo?"},
]
input_text = tokenizer.apply_chat_template(chat, tokenize=False)

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

print("Generating...")

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **input_ids,
        # max_new_tokens=26,
        do_sample=False,
        # top_p=0.95,
        # top_k=50,
        temperature=0.4,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
print(generated_text)