from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from dotenv import load_dotenv
import os
load_dotenv()

model_path =  os.getenv('BASE_MODEL')
data_path = "./data"
target_lora_path = "./lora"

quantize_config = BitsAndBytesConfig(load_in_8bit=True)
# quantize_config = None

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main",
                                             quantization_config=quantize_config)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

model.train()  # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

config.inference_mode = False

# LoRA trainable version of model
model = get_peft_model(model, config)

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
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = TrainingArguments(
    output_dir= "lora",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
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
trainer.save_model(target_lora_path)
tokenizer.save_pretrained(target_lora_path)