from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./base"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=False, revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

chat = [
   {"role": "user", "content": "What is art?"},
   # {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   # {"role": "user", "content": "I'd like to show off how chat templating works!"},
   # {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   # {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

# print(tokenizer.apply_chat_template(chat, tokenize=False))


# print(tokenizer.get_chat_template())

model.eval()

input_text = "Write me a poem about Machine Learning."

input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=250)

print(tokenizer.decode(outputs[0]))


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
