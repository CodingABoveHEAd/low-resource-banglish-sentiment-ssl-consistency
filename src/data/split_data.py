import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/labeled_data.csv")

# rename if needed
df = df.rename(columns={"sentence": "review", "sentiment": "label"})

train, temp = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Data split done!")