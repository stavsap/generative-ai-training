from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

model_path = "./base"
fine_tuned_path = "./trained"
data_path = "./data"


tokenizer = AutoTokenizer.from_pretrained(model_path)


model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="cuda",
                                             trust_remote_code=False,
                                             revision="main")


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

tokenizer.pad_token = tokenizer.eos_token

args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=260,
    logging_steps=1,
    output_dir="trained",
    optim="adamw_8bit",
    seed=42
)

trainer = SFTTrainer(model=model,
                     tokenizer=tokenizer,
                     args=args,
                     dataset_text_field="text",
                     train_dataset=tokenized_data["train"],
                     eval_dataset=tokenized_data["test"],
                     )

trainer.train()
model.save_pretrained(fine_tuned_path)
tokenizer.save_pretrained(fine_tuned_path)
