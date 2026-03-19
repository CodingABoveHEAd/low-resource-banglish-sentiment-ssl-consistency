import torch.nn as nn
from transformers import AutoModel

class SentimentModel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)  #load the pre-trained XLM-RoBERTa model
        self.fc = nn.Linear(self.encoder.config.hidden_size, 2)  #binary classification (positive/negative) embedding to sentiment logits

    def forward(self, input_ids, attention_mask):    #forward pass through the model
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.fc(cls)