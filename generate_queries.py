
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import csv
from tqdm import tqdm

# Model setup
model_path = "/projects/p32408/3b-results-100/checkpoint-17000"
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fine_tuned_model.to(device)

def generate_query(input_text):
    inputs = fine_tuned_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = fine_tuned_model.generate(**inputs, max_length=256, num_return_sequences=1, num_beams=4)
    generated_query = fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_query

input_file = 'data/proc_test_results.csv'
output_file = 'results/output_cp13000_3b.sql'

processed_count = 0

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    next(reader)  # Skip header row
    
    for row in tqdm(reader, desc="Generating queries", unit="query"):
        question = row[0] if len(row) > 0 else ""
        db_id = row[1] if len(row) > 1 else ""
        table_info = row[2] if len(row) > 2 else ""
        
        # Combine information for input
        if table_info:
            processed_input = f"Question: {question} | Database: {db_id} | Schema: {table_info}"
        else:
            processed_input = f"Question: {question} | Database: {db_id}"
        
        # Generate the query
        generated_query = generate_query(processed_input)
        
        # Write to the output file: generated_query, tab, db_id
        outfile.write(f"{generated_query}\t{db_id}\n")
        
        # Increment processed count
        processed_count += 1
        
        # Print progress (optional)
        print(f"Generated: {generated_query}\t{db_id}")

print(f"Query generation complete. {processed_count} entries processed.")
print(f"Results written to {output_file}")

# Count the number of lines in the output file
with open(output_file, 'r', encoding='utf-8') as f:
    line_count = sum(1 for line in f)

print(f"The output file contains {line_count} lines.")