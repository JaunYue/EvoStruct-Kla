from train import run_training
from dataset import train_list, val_list, test_list


if __name__ == "__main__":
    try:
        train_list, val_list, test_list
    except Exception:
        raise RuntimeError("Provide train_list, val_list, test_list or install essential imports.")

    model, train_stats, val_stats, test_stats, preds, targets = run_training(train_list, val_list, test_list)