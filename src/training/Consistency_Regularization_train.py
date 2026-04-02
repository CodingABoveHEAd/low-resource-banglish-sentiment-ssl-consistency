import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from itertools import cycle
import torch.nn.functional as F

from src.data.dataset import BanglishDataset
from src.data.unlabeled_dataset import UnlabeledDataset
from src.models.model import SentimentModel
from src.utils.metrics import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(num_epochs=5, lambda_u=0.5, tau=0.8, T=0.5):

    train_ds = BanglishDataset("data/processed/train.csv")
    val_ds = BanglishDataset("data/processed/val.csv")
    unlabeled_ds = UnlabeledDataset("data/raw/unlabeled_data.csv")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=16, shuffle=True)

    model = SentimentModel().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        loop = tqdm(zip(train_loader, cycle(unlabeled_loader)), total=len(train_loader))

        for l_batch, u_batch in loop:
            optimizer.zero_grad()

            # -----------------------------
            # 🔹 Supervised loss
            # -----------------------------
            input_ids = l_batch["input_ids"].to(device)
            attention_mask = l_batch["attention_mask"].to(device)
            labels = l_batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            sup_loss = loss_fn(outputs, labels)

            # -----------------------------
            # 🔹 Unlabeled consistency
            # -----------------------------
            u_input_ids = u_batch["input_ids"].to(device)
            u_attention_mask = u_batch["attention_mask"].to(device)

            # stochastic forward passes
            logits_w = model(u_input_ids, u_attention_mask)  # weak
            logits_s = model(u_input_ids, u_attention_mask)  # strong

            probs_w = torch.softmax(logits_w.detach(), dim=1)

            # confidence mask
            max_probs, _ = torch.max(probs_w, dim=1)
            mask = (max_probs >= tau).float()

            # sharpening
            probs_w = probs_w ** (1 / T)
            probs_w = probs_w / probs_w.sum(dim=1, keepdim=True)

            log_probs_s = F.log_softmax(logits_s, dim=1)

            consistency_loss = F.kl_div(
                log_probs_s, probs_w, reduction='none'
            ).sum(dim=1)

            # apply mask
            consistency_loss = (consistency_loss * mask).mean()

            # -----------------------------
            # 🔹 Total loss
            # -----------------------------
            loss = sup_loss + lambda_u * consistency_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}]")
            loop.set_postfix(loss=loss.item())

        print(f"\nEpoch {epoch+1} Avg Loss: {total_loss:.4f}")

        acc, f1 = evaluate(model, val_loader)

        # save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "checkpoints/consistency_regularization_pi_model_best_model.pt")
            print(f"Best model saved! F1: {best_f1:.4f}")

    return


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