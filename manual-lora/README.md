# Readme


## Download Data Set form hugging face

Install large file system support for git
```shell
git lfs install

```

Download example data set locally

```shell
git clone https://huggingface.co/datasets/shawhin/shawgpt-youtube-comments
```

cd to merged

```shell
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py -h
python llama.cpp/convert_hf_to_gguf.py merged
python llama.cpp/convert_lora_to_gguf.py -h
python llama.cpp/convert_lora_to_gguf.py --base base lora/checkpoint-30
```