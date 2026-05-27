import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from preprocessing import clean_text

MODEL_NAME = "allegro/herbert-base-cased"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    data_path = os.path.join(BASE_DIR, "BAN-PL_2", "BAN-PL.csv")

    df = pd.read_csv(data_path)

    print("Kolumny:", df.columns)

    # usuń kolumny typu Unnamed
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

    # preprocessing
    df["Text"] = df["Text"].apply(clean_text)

    # multi-label pivot
    df["value"] = 1
    df_multi = df.pivot_table(
        index="Text",
        columns="Reason",
        values="value",
        fill_value=0
    ).reset_index()

    texts = df_multi["Text"].tolist()
    labels = df_multi.drop(columns=["Text"]).values

    return texts, labels


def create_dataset(texts, labels, tokenizer):
    dataset = Dataset.from_dict({
        "text": texts,
        "labels": labels
    })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=256
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset


def train():
    texts, labels = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = create_dataset(texts, labels, tokenizer)

    split = dataset.train_test_split(test_size=0.2)
    train_dataset = split["train"]
    val_dataset = split["test"]

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels[0]),
        problem_type="multi_label_classification"
    )

    # katalog wyników
    results_path = os.path.join(BASE_DIR, "results")
    model_path = os.path.join(BASE_DIR, "model")

    training_args = TrainingArguments(
        output_dir=results_path,
        learning_rate=0.01,
        per_device_train_batch_size=8,
        num_train_epochs=3,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    
    print("Start treningu...")
    trainer.train()
    print("Trening zakończony")


    # zapis modelu
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"Model zapisany w: {model_path}")


if __name__ == "__main__":
    train()