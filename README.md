# Dialectify

Dialectify is a machine learning-based tool that detects and converts different dialects of English (such as American and British English) based on user input. It uses NLP (Natural Language Processing) models to recognize dialect differences and suggests auto-fixing and error-checking based on the dialect specified by the user.

## Features

- **Dialect Detection**: Detect whether the input text is in American or British English.
- **Dialect Conversion**: Convert between different dialects based on the user's preference.
- **Error Checking**: Identify and fix spelling differences between dialects.

## Tech Stack

- **Language**: Python
- **Framework**: Hugging Face's Transformers
- **Machine Learning Model**: DistilBERT (Distilled version of BERT)
- **Dependencies**: 
  - `transformers`
  - `torch`
  - `scikit-learn`
  - `pandas`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sseth30/Dialectify.git
cd Dialectify

### 2. Set up a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### 3. Install dependencies
pip install -r requirements.txt

### 4. Running the code
To train the model, run the following:
python3 src/ml/train_model.py

To predict the dialect of a sentence:
python3 src/ml/predict.py

### For example:

Sentence: I love the color of your car.
Predicted Dialect: American English

Sentence: I love the colour of your car.
Predicted Dialect: British English
