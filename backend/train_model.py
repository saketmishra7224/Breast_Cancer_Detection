from app.ml.train import train_and_save_best


def main():
    print("Training models (this may take a few minutes). Output -> backend/artifacts/")
    best_name, metrics = train_and_save_best()
    print(f"Best model: {best_name}")
    print("Metrics summary:")
    for m in metrics:
        print(f"- {m['model']}: ROC-AUC={m['roc_auc']:.3f}, F1={m['f1']:.3f}, Acc={m['accuracy']:.3f}")


if __name__ == '__main__':
    main()
