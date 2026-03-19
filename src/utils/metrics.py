from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, f1