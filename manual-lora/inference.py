from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./base"

# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=False, revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

print(tokenizer.apply_chat_template(chat, tokenize=False))


# print(tokenizer.get_chat_template())

# model.eval()
#
# inputs = tokenizer('What is the largest animal?', return_tensors="pt")
#
# outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"))
#
# print(tokenizer.batch_decode(outputs)[0])

