import os

def process_file(input_filename, output_filename):
    with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
        for line in input_file:
            # Split the line at the first '|' character
            parts = line.split('|', 1)
            if len(parts) > 1:
                processed_line = parts[1].strip()
                # Ensure the line ends with a semicolon
                if processed_line and not processed_line.endswith(';'):
                    processed_line += ';'
                output_file.write(processed_line + '\n')

    print(f"Processed file saved as {output_filename}")

# Specify the input and output filenames
input_filename = 'results/output_cp13000_3b.sql'  # Replace with your input SQL filename
output_filename = 'processed_queries/processed_output_cp17000_3b.sql' # Replace with your output SQL filename

# Check if the input file exists
if not os.path.exists(input_filename):
    print(f"Error: Input file '{input_filename}' not found.")
else:
    process_file(input_filename, output_filename)