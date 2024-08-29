from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./trained"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=False, revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# https://huggingface.co/docs/transformers/main/en/chat_templating

chat = [
    {"role": "user", "content": "What is art?"},
    {"role": "assistant", "content": "It is whatever you want it to be"},
]

print(tokenizer.apply_chat_template(chat, tokenize=False))

print(tokenizer.get_chat_template())

model.eval()

input_text = "What is Light?"

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=500)

print(tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True))

# import torch
# from transformers import pipeline
#
# pipe = pipeline(
#     "text-generation",
#     model=model_path,
#     device="cuda",  # replace with "mps" to run on a Mac device
# )
#
# text = "Once upon a time,"
# outputs = pipe(text, max_new_tokens=256)
# response = outputs[0]["generated_text"]
# print(response)
