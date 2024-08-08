# Readme


## Required packages

Cuda support:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install peft datasets bitsandbytes
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
```

## Download Data Set form hugging face

Install Large File Storage support for git.

Then initialize it:
```shell
git lfs install
```

Download example data set locally

```shell
git clone https://huggingface.co/datasets/shawhin/shawgpt-youtube-comments
```

Convert to gguf files.

```shell
python llama.cpp/convert_hf_to_gguf.py base
python llama.cpp/convert_lora_to_gguf.py --base base lora
```

## Ollama

```shell
ollama create manual -f Modelfile
ollama run manual
```