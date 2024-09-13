from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

model_path =  os.getenv('BASE_MODEL')
fine_tuned_path = "./trained"
data_path = "./data"


tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=False,
                                             revision="main")

data = load_dataset(data_path)

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["example"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length"
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)
tokenized_data = data
# tokenized_data = data
tokenizer.pad_token = tokenizer.eos_token

args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=20,
    logging_steps=1,
    output_dir=fine_tuned_path,
    optim="adamw_8bit",
    seed=42
)

trainer = SFTTrainer(model=model,
                     tokenizer=tokenizer,
                     args=args,
                     dataset_text_field="example",
                     train_dataset=tokenized_data["train"],
                     eval_dataset=tokenized_data["test"],
                     )

trainer.train()
model.save_pretrained(fine_tuned_path)
tokenizer.save_pretrained(fine_tuned_path)
