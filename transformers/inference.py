from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModelForCausalLM,PeftConfig
from dotenv import load_dotenv
import os
import torch

load_dotenv()

model_path = os.getenv('BASE_MODEL')
adapter_path = "./lora/checkpoint-140"

try:
    # Optionally enable quantization
    quantize_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=False,
        revision="main",
        # quantization_config=quantize_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


    model = PeftModelForCausalLM.from_pretrained(model, adapter_path)
    # model = model.merge_and_unload()
    model.eval()  # Set the model to evaluation mode

    # Prepare input
    chat = [
        {"role": "user", "content": "What is Mojo?"},
    ]
    input_text = tokenizer.apply_chat_template(chat, tokenize=False)

    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    print("Generating...")

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            # max_new_tokens=26,
            do_sample=False,
            # top_p=0.95,
            # top_k=50,
            temperature=0.1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
    print(generated_text)

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    # Clear CUDA cache
    torch.cuda.empty_cache()