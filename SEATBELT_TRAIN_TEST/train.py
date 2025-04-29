"""
    Script for training model on train data

    Set MODEL_ARCHITECTURE_IDX = 0,1,2,3
        (row) 23
    Set directory for final models:
        final_dir = "./models/"
        (row) 31
    Experiment configuration is set
        - sequence length and features type
        - single experiment to train just in main()
        - multiple experiments - train_all_experiments()

"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import MyDataset

from collate_fn import collate_fn
import logging
from hyperparameters import *

MODEL_ARCHITECTURE = ["basic","bidirectional","convolution","attention"]
MODEL_ARCHITECTURE_IDX = 0

final_dir = f"./models_tmp/"

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
log_file_path = os.path.join(logging_dir, 'log.txt')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])
logging.info("Starting training...")

def train(experiment_name):
    train_data_dir = f'../resources/Combined_SB_experiments_split/{experiment_name}/train'
    model_name = f"my_{model_architecture}_{experiment_name}"

    logging.info(f"Experiment name: {experiment_name}")

    # output dirs
    models_dir = "./runs"  # for checkpoints
    train_number = len([d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]) + 1
    train_dir = os.path.join(models_dir, f"train{train_number}")
    os.makedirs(train_dir, exist_ok=True)

    train_dataset = MyDataset(train_data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model = ActionRecognitionLSTM(input_size, hidden_size, num_classes, num_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for keypoints, labels in train_dataloader:
            keypoints, labels = keypoints.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_preds / total_preds * 100
        epoch_time = time.time() - epoch_start_time

        logging.info(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s")

        # checkpoints
        if epoch % 10 == 0 or epoch >= 0.98 * num_epochs:
            model_save_path = os.path.join(train_dir, f"action_recognition_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Model saved at epoch {epoch} to {model_save_path}")

    # final model
    final_model_path = os.path.join(final_dir, model_name)
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")


def train_all_experiments():

    lstm_block_counts = [3,4,5,6]
    # lstm_block_counts = [5]
    features_types = ["A","B","C","D","E"]

    for lstm_block_count in lstm_block_counts:
        for features_type in features_types:
            experiment_name = f"size_{lstm_block_count}_features{features_type}"
            train(experiment_name)

    logging.info("Training completed.")

def main():
    experiment = "size_3_featuresB"
    # experiment = "size_5_featuresE"
    train(experiment)

if __name__ == "__main__":

    main()
    # train_all_experiments()
    pass