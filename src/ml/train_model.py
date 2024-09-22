from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
from datasets import load_dataset
import torch

# Mapping for dialects (string labels to integer labels)
dialect_mapping = {
    "American": 0,
    "British": 1
}

def train_dialect_detector():
    # Load dataset
    dataset = load_dataset('csv', data_files={'train': 'data/train.csv', 'test': 'data/test.csv'})

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Tokenize the input and filter out any rows with invalid or missing dialects
    def tokenize_function(examples):
        valid_examples = {"text": [], "labels": []}
        
        for text, dialect in zip(examples["text"], examples["dialect"]):
            if dialect in dialect_mapping:
                valid_examples["text"].append(text)
                valid_examples["labels"].append(dialect_mapping[dialect])
        
        # Tokenize the valid examples
        tokenized_inputs = tokenizer(valid_examples['text'], padding="max_length", truncation=True)
        tokenized_inputs["labels"] = valid_examples["labels"]
        
        return tokenized_inputs

    # Map the tokenize function to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # Set dataset format to torch tensors
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer to a directory
    model.save_pretrained('./models/dialect_detector')  # Save in a directory, not a file
    tokenizer.save_pretrained('./models/dialect_detector')  # Save tokenizer in the same directory
    
if __name__ == "__main__":
    train_dialect_detector()
