from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

def intialize_bert_model(path='bert-base-uncased', num_labels=3):
    # Intialized_model
    model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)    
    
    # 0=negative, 1=positive, 2=neutral
    # Modify label names according to datasets
    model.config.id2label = {
        0: "negative",
        1: "positive",
        2: "neutral"
    }

    model.config.label2id = {
        "negative": 0,
        "positive": 1,
        "neutral": 2
    }
    
    # Intialized tokenizer
    tokenizer = BertTokenizer.from_pretrained(path)
    
    return model, tokenizer