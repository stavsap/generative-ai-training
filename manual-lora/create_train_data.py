import os
import random

import pandas as pd
from transformers import AutoTokenizer

model_path = "./base"
data_target_path = "./data"
num_samples = 1000


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


create_folder(data_target_path)

def get_random_entry() -> (str, str):
    qa_pairs = [
        ("What is the function of DNA?",
         """DNA (Deoxyribonucleic Acid) has several key functions:
         1. Storage of genetic information
         2. Transmission of genetic information to offspring
         3. Directing the synthesis of proteins
         4. Regulation of gene expression
         DNA's structure allows it to replicate and pass on genetic traits."""),

        ("How does photosynthesis work?",
         """Photosynthesis is a complex process that can be summarized in these steps:
         1. Light absorption by chlorophyll
         2. Excitation of electrons in chlorophyll
         3. Electron transport chain and ATP production
         4. Carbon fixation in the Calvin cycle
         5. Production of glucose from CO2 and water
         This process converts light energy into chemical energy stored in glucose.""")
    ]
    return random.choice(qa_pairs)


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# https://huggingface.co/docs/transformers/main/en/chat_templating

# The rest of the code remains the same

# Generate dataset
questions = []
answers = []
examples = []

for _ in range(num_samples):
    q, a = get_random_entry()
    questions.append(q)
    answers.append(a)

    chat = [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]

    example = tokenizer.apply_chat_template(chat, tokenize=False)
    examples.append(example)

# Create DataFrame
df = pd.DataFrame({
    'question': questions,
    'answer': answers
})

dfEx = pd.DataFrame({
    'example': examples,
})

# Save as Parquet file
df.to_parquet(data_target_path+'/qa.parquet')
dfEx.to_parquet(data_target_path+'/examples.parquet')

# Read back the Parquet file to verify
# read_df = pd.read_parquet('biology_qa_dataset.parquet')

# # Display first few rows
# print(read_df.head())
#
# # Display info about the dataset
# print(read_df.info())
