from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

model_path = "./base"
data_path = "./data"
target_lora_path = "lora"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=False, revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

model.train()  # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
# model = prepare_model_for_kbit_training(model)

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
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)

# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
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
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)

model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
trainer.train()

model.config.use_cache = True

trainer.save_model(target_lora_path)