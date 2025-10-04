from transformers import BertTokenizer

def load_data():
    from datasets import load_dataset
    dataset = load_dataset("mteb/amazon_massive_scenario", "en")
    print(dataset)
    return dataset

def load_tokenizer(args): 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
    return tokenizer