import pandas as pd
import os 
import torch
import sqlite3
import re
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ['HF_HOME'] = '/projects/p32408/cache'

# Set cache directory - this is just for Quest
os.environ['HF_HOME'] = '/projects/p32408/cache'

def load_csv_to_dataset(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, quotechar='"', escapechar='\\')
    
    # Convert the dataframe to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset

# File paths - formatted_train_data, formatted_validation_data will be produced by data_processing.ipynb
train_file_path = "data/formatted_train_data.csv"
validation_file_path = "data/formatted_validation_data.csv"

# Load the datasets
train_dataset = load_csv_to_dataset(train_file_path)
validation_dataset = load_csv_to_dataset(validation_file_path)

# Combine into a DatasetDict
spider_dataset = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset
})

# Print some information about the dataset
print("Dataset created successfully!")
print(f"Number of training examples: {len(spider_dataset['train'])}")
print(f"Number of validation examples: {len(spider_dataset['validation'])}")

# Display the first example from the training set
print("\nFirst example from the training set:")
print(spider_dataset['train'][0])

# Commented out code is for fine tuning t5 (not checkpoint)

# # Load the SPIDER dataset
# dataset = load_dataset("spider", cache_dir="/projects/p32408/cache/new")

# Load the T5 tokenizer and model
# checkpoint = "google-t5/t5-3b"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="/projects/p32408/cache")
# model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir="/projects/p32408/cache")

# for tuning from existing checkpoint 

checkpoint_path = "./3b-results-100/checkpoint-12500"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

# #device=torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Schema serialization function
def serialize_schema(db_id, table_info, schema_serialization_type="peteshaw",
                     schema_serialization_with_db_id=True,
                     schema_serialization_with_db_content=True):
    if schema_serialization_type == "peteshaw":
        schema_str = []
        if schema_serialization_with_db_id:
            schema_str.append(f"Database: {db_id}")
        if schema_serialization_with_db_content:
            schema_str.append(table_info)
        return " | ".join(schema_str)
    else:
        raise ValueError(f"Unknown schema serialization type: {schema_serialization_type}")

def preprocess_function(examples):
    questions = examples["question"]
    db_ids = examples["db_id"]
    table_infos = examples["table_info"]

    schemas = [
        serialize_schema(db_id, table_info,
                         schema_serialization_type="peteshaw",
                         schema_serialization_with_db_id=True,
                         schema_serialization_with_db_content=True)
        for db_id, table_info in zip(db_ids, table_infos)
    ]

    inputs = [f"Question: {q} | {s}" for q, s in zip(questions, schemas)]
    targets = [f"{db_id} | {query}" for db_id, query in zip(db_ids, examples["query"])]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Apply the preprocessing to your dataset
tokenized_datasets = spider_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=spider_dataset["train"].column_names
)

print(tokenized_datasets)


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./3b-results-100",
    run_name="g5-large-db-id",
    do_train=True,
    do_eval=False,
    evaluation_strategy="no",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    learning_rate=5e-5,
    num_train_epochs=500,
    adafactor=True,
    lr_scheduler_type="constant",
    warmup_ratio=0.0,
    warmup_steps=0,
    save_steps=500,
    save_total_limit=50,
    fp16=False,
    predict_with_generate=True,
    generation_max_length=128,
    logging_strategy="steps",
    logging_steps=5,
    logging_first_step=True,
    report_to=["wandb"],
    metric_for_best_model="exact_match",
    greater_is_better=True,
    seed=1,
    overwrite_output_dir=True,
    label_smoothing_factor=0.0,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    
)

trainer.train(resume_from_checkpoint=checkpoint_path)

