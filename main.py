import argparse
from train import train
from predict import predict

def main():
    parser = argparse.ArgumentParser(description="Toxic Comment Detector")

    parser.add_argument("--mode", type=str, default="train", required=True, help="train / predict")

    parser.add_argument("--text", type=str, help="tekst do analizy")

    args = parser.parse_args()

    if args.mode == "train":
        print("Trenowanie modelu")
        train()

    elif args.mode == "predict":
        if not args.text:
            print("Podaj tekst (--text)")
            return

        print("Analiza tekstu...")
        probs = predict(args.text)

        for i, p in enumerate(probs):
            print(f"Klasa {i}: {p:.3f}")

    else:
        print("Nieznany tryb")

if __name__ == "__main__":
    main()