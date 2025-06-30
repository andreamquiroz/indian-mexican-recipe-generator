# generate_recipe.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-fusion-recipes")
model = GPT2LMHeadModel.from_pretrained("./gpt2-fusion-recipes").to(device)

# Prompt format — your input starts like this
prompt1 = """
Title: Tikka Tacos
Ingredients:
- Chicken
- Tikka masala sauce
- Tortillas
Instructions:
"""
prompt = """
Title: Chicken Tikka Tacos
"""

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(
    **inputs,
    max_length=300,
    temperature=0.9,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
