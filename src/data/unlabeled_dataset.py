from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class UnlabeledDataset(Dataset):
    def __init__(self, file, model_name="xlm-roberta-base"):
        self.df = pd.read_csv(file)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.texts = self.df["review"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        return {key: val.squeeze() for key, val in encoding.items()}