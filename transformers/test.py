from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

# From: https://huggingface.co/docs/transformers/training

# Model: https://huggingface.co/google-bert/bert-base-cased
model_path = "google-bert/bert-base-cased"

# DB: https://huggingface.co/datasets/Yelp/yelp_review_full
data_path = "yelp_review_full"

fine_tuned_path = "./trained"

dataset = load_dataset(data_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

training_args = TrainingArguments(output_dir=fine_tuned_path, eval_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

tokenizer.save_pretrained(fine_tuned_path)