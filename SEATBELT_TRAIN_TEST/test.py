"""
    Script for testing model on test/val data

    Set MODEL_ARCHITECTURE_IDX = 0,1,2,3
        (row) 29
    Set output_file_path = "./results.csv"
        (row) 31
    Experiment to test has to be specified based on name
        - sequence length and features type
        - single experiment test just in main()
        - multiple experiments - test_all_experiments()

"""

import csv
import os
import time
import torch
from torch.utils.data import DataLoader
from datasets import MyDataset
from collections import defaultdict

from collate_fn import collate_fn
import logging
from hyperparameters import *


MODEL_ARCHITECTURE = ["basic","bidirectional","convolution","attention"]
MODEL_ARCHITECTURE_IDX = 2

csv_file = "./results/test_results3.csv"

if MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "basic":
    from model_basic import ActionRecognitionLSTM
elif MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "bidirectional":
    from model_bidirectional import ActionRecognitionLSTM
elif MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "convolution":
    from model_convolution import ActionRecognitionLSTM
elif MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX] == "attention":
    from model_attention import ActionRecognitionLSTM
else:
    raise Exception("Unknown model")

model_architecture = f"model_{MODEL_ARCHITECTURE[MODEL_ARCHITECTURE_IDX]}"

# logging setup
logging_dir = "./logs/"
os.makedirs(logging_dir, exist_ok=True)
log_file_name = f"test_log_{model_architecture}.txt"
log_file_path = os.path.join(logging_dir, log_file_name)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])
logging.info(f"Architecture: {model_architecture}")
logging.info("Starting testing...")



def results_in_csv(experiment_name, tp, tn, fp, fn, test_accuracy, precision, recall, f1_score):

    size = int(experiment_name.split('_')[1])  # size_3 -> 3
    features = experiment_name.split('_')[2].replace("features", "")  # featuresA -> A

    header = ["architecture", "sequence_length", "features", "tp", "tn", "fp", "fn","accuracy", "precision", "recall", "f1_score"]

    new_row = {
        "architecture": model_architecture,
        "sequence_length": size,
        "features": features,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(test_accuracy, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_row)


def test(experiment_name):
    test_data_dir = f'../resources/Combined_SB_experiments_split/{experiment_name}/test'
    model_path = f"./models/my_{model_architecture}_{experiment_name}"

    logging.info(f"Experiment name: {experiment_name}")

    test_dataset = MyDataset(test_data_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionLSTM(input_size, hidden_size, num_classes, num_layers).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    total_preds = 0
    correct_preds = 0
    test_start_time = time.time()

    # confusion matrix counters
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    all_classes = set()
    total_per_class = defaultdict(int)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for keypoints, labels in test_dataloader:
            keypoints, labels = keypoints.to(device), labels.to(device)

            lengths = torch.tensor([len(seq[seq.sum(dim=1) != 0]) for seq in keypoints], dtype=torch.long)
            lengths = lengths.cpu().to(torch.int64)

            outputs = model(keypoints)
            _, predicted = torch.max(outputs, 1)

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for pred, true in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                all_classes.update([pred, true])
                total_per_class[true] += 1
                if pred == true:
                    true_positives[true] += 1
                else:
                    false_positives[pred] += 1
                    false_negatives[true] += 1

    test_accuracy = correct_preds / total_preds * 100
    test_time = time.time() - test_start_time

    # test results
    precisions = []
    recalls = []
    f1s = []
    true_negatives = {}
    for cls in all_classes:
        tp = true_positives[cls]
        fp = false_positives[cls]
        fn = false_negatives[cls]
        tn = total_preds - (tp + fp + fn)
        true_negatives[cls] = tn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1s) / len(f1s)

    logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
    logging.info(f"Macro Precision: {macro_precision:.4f}")
    logging.info(f"Macro Recall: {macro_recall:.4f}")
    logging.info(f"Macro F1 Score: {macro_f1:.4f}")
    logging.info(f"Testing completed in {test_time:.2f} seconds.")

    total_tp = sum(true_positives.values())
    total_fp = sum(false_positives.values())
    total_fn = sum(false_negatives.values())
    total_tn = sum(true_negatives.values())

    results_in_csv(
        experiment_name, total_tp, total_tn, total_fp, total_fn,
        test_accuracy, macro_precision, macro_recall, macro_f1
    )


def test_all_experiments():

    lstm_block_counts = [3,4,5,6]
    # lstm_block_counts = [5]
    features_types = ["A","B","C","D","E"]

    for lstm_block_count in lstm_block_counts:
        for features_type in features_types:
            experiment_name = f"size_{lstm_block_count}_features{features_type}"
            test(experiment_name)

    logging.info("Testing completed.")

def main():
    experiment = "size_5_featuresE"
    test(experiment)

if __name__ == "__main__":
    main()
    # test_all_experiments()
    pass