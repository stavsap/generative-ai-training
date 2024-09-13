import os
import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

model_path = os.getenv('BASE_MODEL')
source_data ="./data_raw"
target_data = "./data"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

def create_folder(path):
    """
    Create a folder if it doesn't exist, otherwise delete it and its contents.

    Args:
        path (str): Path to the folder.
    """
    # Check if the folder exists
    if os.path.exists(path):
        # If it exists, delete it and all its contents
        print(f"Deleting existing folder at {path}...")
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        # Remove the parent directory
        os.rmdir(path)

    print(f"Creating new folder at {path}...")
    os.makedirs(path, exist_ok=True)

create_folder(target_data)

def load_data_frame(dp):
    return pd.read_parquet(dp)

def create_conversation2(sample):
    user_message = sample["question"]
    assistant_replay = sample["answer"]
    # return f'''<s>[INST]{user_message}[/INST] {assistant_replay}</s>'''
    return f'''{user_message} {assistant_replay}'''

def create_conversation(sample):
    user_message = sample["question"]
    assistant_replay = sample["answer"]

    return {
        "messages": [
            # {"role": "system", "content": "you are an ai assistant"},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_replay},
        ]
    }

def prep(source_path, target_path, tokenizer):
    df = load_data_frame(source_path)
    df['conversion'] = df.apply(create_conversation, axis=1)

    # https://huggingface.co/docs/transformers/main/en/chat_templating
    text_samples = df['conversion'].apply(lambda x: tokenizer.apply_chat_template(x['messages'],
                                                                                  tokenize=False,
                                                                                  add_generation_prompt=True,
                                                                                  return_tensors="pt"))
    # text_samples = df['conversion']
    test_ds = {
        "example": [t for t in text_samples],
    }

    test_data_set = Dataset.from_dict(test_ds)
    test_data_set.to_parquet(target_path)

for root, dirs, files in os.walk(source_data, topdown=True):
    for file in files:
        prep(os.path.join(root, file), os.path.join(target_data, file), tokenizer)
