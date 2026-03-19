import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.data.preprocessing import clean_text

class BanglishDataset(Dataset):
    def __init__(self, file, model_name="xlm-roberta-base"):
        import pandas as pd
        self.df = pd.read_csv(file)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.texts = self.texts = [clean_text(t) for t in self.df["review"].tolist()]
        self.labels = self.df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],    #tokenize the text
            padding="max_length",  #all sequence become 64 tokens long
            truncation=True,   #truncate if longer than 64 tokens
            max_length=64,
            return_tensors="pt"  #return PyTorch tensors
        )

        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item