# Manual Lora 

Train a lora, convert to gguf and create ollama model.


## Required packages

Cuda support:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install peft datasets bitsandbytes optimum auto-gptq
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