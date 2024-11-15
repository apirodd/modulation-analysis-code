import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
import torch
import ast
import re

# Function to balance parentheses
def balance_parentheses(expression):
    open_count = expression.count('(')
    close_count = expression.count(')')
    
    # Add closing parentheses if necessary
    while open_count > close_count:
        expression += ')'
        close_count += 1
    
    return expression

# Function to verify syntax
def is_valid_syntax(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# Define the device as CPU
device = torch.device('cpu')

# Load the dataset
df = pd.read_csv('random_modulations.csv')
texts = df['Formula'].values
text_data = "\n".join(texts)

# Create a Hugging Face dataset
dataset = Dataset.from_dict({"text": text_data.split('\n')})

# Initialize the tokenizer and GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Add labels equal to inputs for loss calculation
def add_labels(batch):
    batch["labels"] = batch["input_ids"].copy()  # Use inputs as labels
    return batch

tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# Define training parameters
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    use_cpu=True,  # Updated to use 'use_cpu'
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not using masked language modeling (MLM)
    )
)

# Train the model
trainer.train()

# Function to filter undefined variables
def filter_undefined_variables(expression, defined_vars):
    # Create a regex pattern for variable names (e.g., I(t), Q(t), etc.)
    pattern = r'\b([A-Za-z_][A-Za-z0-9_]*\(\s*.*?\))\b'
    
    # Find all variables in the expression
    found_vars = re.findall(pattern, expression)
    
    # Check against defined_vars
    for var in found_vars:
        var_name = var.split('(')[0]  # Get the variable name (e.g., I from I(t))
        if var_name not in defined_vars:
            # Replace undefined variable with a default value or remove it
            expression = expression.replace(var, '0')  # Replace with 0 or any default value
            
    return expression

# Function to generate new modulations
def generate_modulation(model, tokenizer, prompt, max_length=100, num_return_sequences=10, temperature=0.8):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  # Use the device here
    attention_mask = torch.ones(input_ids.shape, device=device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=num_return_sequences, temperature=temperature, do_sample=True)
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    
    # Filter undefined variables for each generated text
    defined_vars = ['I', 'Q', 'd', 'A', 'phi', 'f_c', 'm', 'k', 'n', 'T', 'A_c', 'phi_c']  # Add all defined variables here
    filtered_texts = [filter_undefined_variables(text, defined_vars) for text in generated_texts]
    
    return filtered_texts

# Function to check and correct syntax
def check_and_correct_syntax(modulation):
    modulation = balance_parentheses(modulation)  # Balance parentheses
    return modulation, is_valid_syntax(modulation)  # Return the corrected modulation and validity

# Generate new modulations
seed_text = "I(t) * np.cos(2 * np.pi * f_c * t) - Q(t) * np.sin(2 * np.pi * f_c * t)"
generated_modulations = generate_modulation(model, tokenizer, seed_text)

# Validate and filter generated modulations
valid_modulations = []
for modulation in generated_modulations:
    corrected_modulation, is_valid = check_and_correct_syntax(modulation)
    if is_valid:
        valid_modulations.append(corrected_modulation)
    else:
        print(f"Invalid Modulation: {corrected_modulation}")

# Print valid modulations
for i, modulation in enumerate(valid_modulations):
    print(f"Valid Modulation {i+1}:", modulation)
