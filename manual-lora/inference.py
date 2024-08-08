from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./base"

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="cuda",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

print(tokenizer.get_chat_template())

# model.eval()
#
# inputs = tokenizer('What is the largest animal?', return_tensors="pt")
#
# outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"))
#
# print(tokenizer.batch_decode(outputs)[0])

