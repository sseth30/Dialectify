from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

def load_model():
    # Load model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained('./models/dialect_detector')
    tokenizer = DistilBertTokenizer.from_pretrained('./models/dialect_detector')
    return model, tokenizer

def predict_dialect(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    # Get softmax probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get the predicted class
    predictions = torch.argmax(outputs.logits, dim=-1)
    
    pred_class = "American English" if predictions.item() == 0 else "British English"
    
    # Print the probabilities for both classes
    print(f"Probabilities: American English: {probabilities[0][0].item()}, British English: {probabilities[0][1].item()}")
    
    return pred_class

if __name__ == "__main__":
    model, tokenizer = load_model()
    
    # Examples to test
    test_sentences = [
        "I love the color of your car.",  # Clear American spelling
        "The neighbours are quite friendly.",  # British spelling with "neighbours"
        "Let's meet in the car park.",  # British phrase ("car park" vs "parking lot")
        "We need to check the lift in the building.",  # British ("lift" vs "elevator")
        "I can't wait to have some french fries.",  # American phrase
        "Mum and I are going to the shop.",  # British (with "Mum" instead of "Mom")
        "The programme starts at 7 pm.",  # British spelling ("programme")
        "The apartment has a beautiful balcony.",  # American spelling
        "I prefer chips with my burger.",  # Could be tricky (Brits say "chips" for fries, Americans say "chips" for crisps)
        "It's time to get petrol for the car.",  # British phrase ("petrol" vs "gas")
        "I'm going to the chemist to pick up my prescription.",  # British (pharmacy in the US)
        "The theater was amazing!",  # American spelling ("theater" vs "theatre")
        "She ate a biscuit with her tea.",  # British ("biscuit" is a cookie in the US)
        "He seemed to be the center of attention.",
        "The flight from Delhi to Atlanta was quite long but comfortable.",
        "The bloke that was sitting next to me was very funny."
    ]

    # Test each sentence
    for sentence in test_sentences:
        dialect = predict_dialect(sentence, model, tokenizer)
        print(f"Sentence: {sentence}")
        print(f"Predicted Dialect: {dialect}\n")
