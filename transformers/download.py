from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os
load_dotenv()

token = os.getenv('HF_TOKEN')

models =["meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-2b-it","google/gemma-2-9b-it","TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"]

if token is "":
    token = None

for model in models:
    snapshot_download(repo_id=model,
                      local_dir_use_symlinks=False,
                      revision="main",
                      token=token)

