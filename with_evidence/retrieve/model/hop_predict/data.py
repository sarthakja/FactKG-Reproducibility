import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

#This class defines the training dataset.    
class FactkGDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df_data = df
        self.tokenizer = tokenizer

    # This method defines what is returned for a single element in the training dataset
    # This method takes the index of the element in the dataset as input and returns a triple
    # of tokenized output, attention mask and the (ground truth hop-1)
    def __getitem__(self, index):
        sentence1 = self.df_data.loc[index, 'inputs']
        encoded_dict = self.tokenizer.encode_plus(
                    sentence1,
                    add_special_tokens = True,      
                    max_length = 256,           
                    pad_to_max_length = True,
                    truncation=True,
                    return_attention_mask = True,   
                    return_tensors = 'pt',          
               )
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]

        target_ = torch.tensor(self.df_data.loc[index, 'hop'])
        target = target_-1
        sample = (padded_token_list, att_mask, target)

        return sample
    #Returns the no of items in the data
    def __len__(self):
        return len(self.df_data)
    
#This class defines the test dataset
class FactKGTestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df_data = df
        self.tokenizer = tokenizer

    # This method defines what is returned for a single element in the test dataset 
    # The method returns a tuple of (tokenized input, the attention mask, and a concatenation of
    # sentence and the list of entities in the sentence). For tokenized input, the input is a
    # concatenation of sentence and the list of entities in the sentence 
    def __getitem__(self, index):
        sentence1 = self.df_data.loc[index, 'inputs']

        encoded_dict = self.tokenizer.encode_plus(
                    sentence1, 
                    add_special_tokens = True,      
                    max_length = 256,           
                    pad_to_max_length = True,
                    return_attention_mask = True,   
                    truncation=True,
                    return_tensors = 'pt',          
               )
        
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        sample = (padded_token_list, att_mask, sentence1)  # Returning raw sentence here
        
        return sample
    
    # This method returns the no of items in the test dataset
    def __len__(self):
        return len(self.df_data)

# This method creates the training, validation, and test datasets for the hop prediction task. 
# Training and validation datasets are created by calling the FactkGDatset method and test datset
# is created by calling the FactKGTestDataset method
def create_datasets(train_path, dev_path, test_path,model_name):
    model_name = model_name
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    train_json = pd.read_json(train_path)
    dev_json = pd.read_json(dev_path)
    test_json = pd.read_json(test_path)

    train_data = FactkGDataset(train_json, tokenizer)
    val_data = FactkGDataset(dev_json, tokenizer)
    test_data = FactKGTestDataset(test_json, tokenizer)

    return train_data, val_data, test_data
