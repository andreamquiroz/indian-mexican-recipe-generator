from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re

model_name = "PinchuPanda/Recipe-fusion-gen"
model_dir = "./gpt2-fusion-recipes"

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)


def generate_recipe(dish_name):
    prompt = f"Title: {dish_name}\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=300,
        temperature=0.9,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Truncate after second double line break
    parts = re.split(r'\n\s*\n', text)
    if len(parts) > 2:
        text = '\n\n'.join(parts[:2])

    print(text)
