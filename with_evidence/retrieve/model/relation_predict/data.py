#This file is for handling the data for relation prediction.
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import pytorch_lightning as pl
import pickle

# This class defines methods that dictate the formation of train, validation and test batches
# This class inherits from pl.LightningDataModule and redefines some of its methods to fit
# in the context of relation prediction 
class FactKGRelationDataModule(pl.LightningDataModule):
  # setting the class properties
  #Arguments:
  # relations: The list of all possible relations in the knowledge graph
  # tokenizer: The tokenizer used to tokenize text before feeding into the language model
  # data: The dataset dictionary that was formed in the data_preprocess.py file
  # batch_size: used to set batch size in training, validation, and test data
  # max_input_len: The max_input_len parameter in tokenizer 

  #Returns: Nothing
  def __init__(self, relations, tokenizer, data, batch_size=16, max_input_len=512):
    super().__init__()
    self.relations = relations
    self.tokenizer = tokenizer
    self.data = data
    self.batch_size = batch_size
    self.max_input_len = max_input_len
    self.mlb = MultiLabelBinarizer()
    self.mlb.classes = self.relations 
    
  def setup(self, stage):
    pass
  
  # This method is used to form a batch for the training dataset of relation prediction
  def train_dataloader(self):
    return torch.utils.data.DataLoader(
        self.data["train"],
        batch_size=self.batch_size,
        shuffle=True,
        collate_fn=self._collate_fn
    )
  # This method is used to form a batch for the validation dataset of relation prediction
  def val_dataloader(self):
    return torch.utils.data.DataLoader(
        self.data["dev"],
        batch_size=self.batch_size * 2,
        shuffle=False,
        collate_fn=self._collate_fn
    )

  # This method is used to form a batch for the test dataset of relation prediction
  def test_dataloader(self):
    return torch.utils.data.DataLoader(
        self.data["test"],
        batch_size=self.batch_size * 2,
        shuffle=False,
        collate_fn=self._collate_fn_test
    )

  # Defines what does each element of a batch consists of in the test dataset
  def _collate_fn_test(self, batch):
    input_texts = [x["inputs"] for x in batch]
    model_inputs = self.tokenizer(
        input_texts,
        max_length=self.max_input_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return model_inputs, input_texts 

  # Defines what does each element of a batch consists of in the training and validation dataset
  def _collate_fn(self, batch):
    input_texts = [x["inputs"] for x in batch]
    answers = [[_ans if _ans in self.relations else "Unknown" for _ans in x["relation"]] for x in batch]
    answers = self.mlb.fit_transform(answers)
    label_ids = torch.FloatTensor(answers)
    model_inputs = self.tokenizer(
        input_texts,
        max_length=self.max_input_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return model_inputs, label_ids  
