import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from itertools import cycle

from src.data.dataset import BanglishDataset
from src.data.unlabeled_dataset import UnlabeledDataset
from src.models.model import SentimentModel
from src.utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():

    train_ds = BanglishDataset("data/processed/train.csv")
    val_ds = BanglishDataset("data/processed/val.csv")
    unlabeled_ds = UnlabeledDataset("data/raw/unlabeled_data.csv")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=16, shuffle=True)

    model = SentimentModel().to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    lambda_u = 0.5  # consistency weight

    best_f1 = 0.0

    for epoch in range(3):
        model.train()
        total_loss = 0

        loop = tqdm(zip(train_loader, cycle(unlabeled_loader)), total=len(train_loader))

        for l_batch, u_batch in loop:

            optimizer.zero_grad()

            # Supervised loss
            input_ids = l_batch["input_ids"].to(device)
            attention_mask = l_batch["attention_mask"].to(device)
            labels = l_batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            sup_loss = loss_fn(outputs, labels)

            # Consistency loss
            u_input_ids = u_batch["input_ids"].to(device)
            u_attention_mask = u_batch["attention_mask"].to(device)

            outputs1 = model(u_input_ids, u_attention_mask)
            outputs2 = model(u_input_ids, u_attention_mask)

            # detach one branch (stabilizes training)
            p1 = torch.softmax(outputs1.detach(), dim=1)
            p2 = torch.softmax(outputs2, dim=1)

            consistency_loss = torch.mean((p1 - p2) ** 2)

            # Total loss
            loss = sup_loss + lambda_u * consistency_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        acc, f1 = evaluate(model, val_loader)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "consistency_regularization_pi_model_best_model.pt")
            print(f"Best model saved! (F1: {best_f1:.4f})")

        model.train()


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
