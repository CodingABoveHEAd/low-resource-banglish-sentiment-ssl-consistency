import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import BanglishDataset
from src.models.model import SentimentModel
from src.utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():

    train_ds = BanglishDataset("data/processed/train.csv")
    val_ds = BanglishDataset("data/processed/val.csv")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = SentimentModel().to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_acc = 0.0

    history = [] 

    for epoch in range(3):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        acc, f1 = evaluate(model, val_loader)

        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": acc,
            "f1": f1
        })

        if (f1 > best_f1) or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f"Best model saved at epoch {epoch+1} (F1: {f1:.4f}, Acc: {acc:.4f})")

    torch.save(model.state_dict(), "checkpoints/last_model.pt")

    return history


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc, f1 = compute_metrics(preds, labels)
    print(f"Validation Accuracy: {acc:.4f}, F1: {f1:.4f}")

    return acc, f1