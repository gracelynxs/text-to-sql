# Text-to-SQL 

## Project Structure

- `data_processing.ipynb`: Jupyter notebook for data preprocessing
- `model_training.py`: Script for training the model
- `generate_queries.py`: Script to generate queries using the trained model
- `process_output.py`: Script to process model output for evaluation
- `example.slurm`: Sample SLURM script for job submission on Quest
- processed_queries: folder containing some final outputs 

## Setup and Requirements

1. Download the Spider 1.0 dataset (required for data processing)
2. Install required dependencies (requirements.txt)

## Usage

### Data Processing

Run `data_processing.ipynb` to produce formatted train/validation CSV files for model input (requires downloading Spider 1.0 dataset)

### Model Training

Run `model_training.py` to train the model. By default, it trains from a checkpoint, but you can modify it to train from a HuggingFace model.

### Generating Queries

After training, use `generate_queries.py` to generate SQL queries

### Processing Output

Process the generated queries for evaluation using [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) by running `process_output.py` 

### Running on SLURM

An example SLURM script (`example.slurm`) is provided for submitting jobs to Quest.

## Fine-tuned model

Latest fine-tuned model: approximately 315 epochs. Match accuracy: 54%, execution accuracy: 57% - download [here](https://drive.google.com/file/d/1EJujDLZ1YJwdwkKCEcTsQeH5RfLxAw7S/view?usp=sharing)
