import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

model_path = "./base"
fine_tuned_path = "./trained"
data_path = "./data/train.parquet"


def load_data_frame(dp):
    return pd.read_parquet(dp)


def create_conversation(sample):
    user_message = sample["question"]
    assistant_replay = sample["answer"]

    return {
        "messages": [
            {"role": "system", "content": "you are an ai assistant"},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_replay},
        ]
    }

tokenizer = AutoTokenizer.from_pretrained(model_path)

df = load_data_frame(data_path)

df['conversion'] = df.apply(create_conversation, axis=1)

# https://huggingface.co/docs/transformers/main/en/chat_templating
text_samples = df['conversion'].apply(lambda x: tokenizer.apply_chat_template(x['messages'],
                                                                              tokenize=False,
                                                                              add_generation_prompt=True,
                                                                              return_tensors="pt"))

dataset = {
    "text": [t for t in text_samples],
}

ds = Dataset.from_dict(dataset)

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=False, revision="main")

args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=60,
    logging_steps=1,
    output_dir="trained",
    optim="adamw_8bit",
    seed=42
)

trainer = SFTTrainer(model=model,
                     tokenizer=tokenizer,
                     args=args,
                     dataset_text_field="text",
                     train_dataset=ds)

trainer.train()
model.save_pretrained(fine_tuned_path)