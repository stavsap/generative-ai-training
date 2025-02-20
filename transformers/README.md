# Fine-Tuning using the transformers packages.

Fine-tune:
- full
- lora
- qlora

## Required packages

Dependencies install support:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install peft datasets bitsandbytes optimum trl python-dotenv evaluate
```

CPP lib to convert to GGUF

```shell
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

Convert to gguf files (base + adapter).

```shell
python llama.cpp/convert_hf_to_gguf.py base
python llama.cpp/convert_lora_to_gguf.py --base base lora
```

Convert merged to gguf file.

```shell
python llama.cpp/convert_hf_to_gguf.py --outtype q8_0 base
```

## Ollama

```shell
ollama create manual:1
ollama run manual:1
```

# Appendix

https://medium.com/the-ai-forum/instruction-fine-tuning-gemma-2b-on-medical-reasoning-and-convert-the-finetuned-model-into-gguf-844191f8d329

https://huggingface.co/google/gemma-7b

Download AutoGPTQ from github: git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
Build from setup.py:
python setup.py build
python setup.py install