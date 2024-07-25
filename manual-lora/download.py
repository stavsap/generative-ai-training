from huggingface_hub import snapshot_download

# model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model_id = "openlm-research/open_llama_3b_v2"

snapshot_download(repo_id=model_id,
                  local_dir="base",
                  local_dir_use_symlinks=False,
                  revision="main")
