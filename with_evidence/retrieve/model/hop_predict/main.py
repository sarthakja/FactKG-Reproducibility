#This file ties together the model.py and data.py files to perform hop prediction

import argparse
import yaml
import os
from torch.utils.data import DataLoader
import torch
import json
from data import create_datasets
from model import HopPredictorManager
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#This method performs the training of the model for hop prediction
#The method also saves the final model(after the training is complete) in 'model.pth' file
def train(config, device):
    torch.cuda.empty_cache()
    train_data, val_data, _ = create_datasets(config['train_data_path'], config['dev_data_path'], config['test_data_path'],config['model_name'])
    train_dataloader = DataLoader(train_data, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(val_data, batch_size=config['VAL_BATCH_SIZE'], shuffle=True, num_workers=os.cpu_count())
    model_manager = HopPredictorManager(config["model_name"], 3, config["LEARNING_RATE"], device)
    for epoch in range(config["NUM_EPOCHS"]):
        print(f'======== Epoch {epoch+1} / {config["NUM_EPOCHS"]} ========')
        print('Training...')
        avg_train_loss = model_manager.train_epoch(train_dataloader, device)
        print(f"Average training loss: {avg_train_loss}")
        print("\nRunning Validation...")
        avg_val_loss, avg_val_accuracy = model_manager.evaluate(val_dataloader, device)
        print(f"Validation Loss: {avg_val_loss}")
        print(f"Validation Accuracy: {avg_val_accuracy}")
    model_manager.save_model('model.pth')

#This method uses the saved model from the train method to obtain predictions on the test dataset
#The predictions are saved in a file named prediction_hop.json
def evaluate(config, model_path, device):
    print('Evaluating...')
    _, _, test_data = create_datasets(config['train_data_path'], config['dev_data_path'], config['test_data_path'],config['model_name'])
    test_dataloader = DataLoader(test_data, batch_size=config['TEST_BATCH_SIZE'], shuffle=False, num_workers=os.cpu_count())
    model_manager = HopPredictorManager(config["model_name"], 3, config["LEARNING_RATE"], device)
    model_manager.load_model(model_path)
    sentences, predictions = model_manager.predict(test_dataloader, device)
    claims_dict = {str(index): claim for index, claim in enumerate(sentences)}
    predictions_dict = {str(index): int(pred) + 1 for index, pred in enumerate(predictions)}
    output_data = {"claims": claims_dict, "predict": predictions_dict}
    with open("predictions_hop.json", "w") as outfile:
        json.dump(output_data, outfile, indent=4)

#The main method for handling the methods above
#First extracts the command line arguments and then based on the mode: "train" or "eval", performs
# training or testing respectively
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == 'train':
        t1 = time.time()
        train(config, device)
        print("Total time taken to train: ", time.time()-t1)
    elif args.mode == 'eval':
        if args.model_path is None:
            raise Exception("Model checkpoint required for evaluation mode.")
        evaluate(config, args.model_path, device)

if __name__ == "__main__":
    main()