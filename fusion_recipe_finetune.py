import os
import json
from datasets import Dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

MODEL_NAME = "gpt2-medium"


def load_tokenizer(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Avoid padding errors
    return tokenizer


def format_recipe(recipe):
    title = recipe.get('title', '').strip().replace('\n', ' ')
    ingredients_raw = recipe.get('ingredients', [])
    if isinstance(ingredients_raw, str):
        try:
            ingredients_list = json.loads(ingredients_raw)
        except json.JSONDecodeError:
            ingredients_list = [ingredients_raw]
    else:
        ingredients_list = ingredients_raw

    ingredients = "\n".join(f"- {item.strip()}" for item in ingredients_list)

    instructions_raw = recipe.get('instructions', [])
    if isinstance(instructions_raw, str):
        try:
            instructions_list = json.loads(instructions_raw)
        except json.JSONDecodeError:
            instructions_list = [instructions_raw]
    else:
        instructions_list = instructions_raw

    instructions = "\n".join(step.strip() for step in instructions_list)

    return f"Title: {title}\nIngredients:\n{ingredients}\nInstructions:\n{instructions}\n\n"


def prepare_dataset(tokenizer, file_path, block_size=512):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted = [format_recipe(r) for r in raw_data]

    tokenized = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=block_size,
    )

    return Dataset.from_dict(tokenized)


def main():
    tokenizer = load_tokenizer(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    train_dataset = prepare_dataset(tokenizer, "combined_fusion_dataset.json")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./gpt2-fusion-recipes",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        num_train_epochs=20,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        learning_rate=5e-5,
        warmup_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model("./gpt2-fusion-recipes")
    tokenizer.save_pretrained("./gpt2-fusion-recipes")


if __name__ == "__main__":
    main()
