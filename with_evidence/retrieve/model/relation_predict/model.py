#This file defines the model for relation prediction 

from transformers import AutoModelForSequenceClassification
import pytorch_lightning as pl
import torch
from functools import reduce
import numpy as np
import random
import pandas as pd


#This class inherits from pl.LightningModule and redefines some of its methods to fit
# in the context of relation prediction 

class FactKGRelationClassifier(pl.LightningModule):
  
  # Arguments:
  # relations: The list of all relations in the knowledge graph
  # model: The transformer based model to use for relation prediction
  # top_k: The no of relations to predict
  # learning_rate: The learning rate of the optimizer

  # Returns: Nothing
  def __init__(self, relations, model, top_k, learning_rate=0.00001):
    super().__init__()
    self.relations = relations
    self.model = model
    self.top_k=top_k
    self.learning_rate = learning_rate


  # This method defines what happens in a single formward pass of the model
  # Arguments: 
  # batch: a single batch of the training data

  # Returns:
  # the loss(a float value) and the logits(2D tensor of dimension (batch size, no of relations in Knowledge graph))
  def forward(self, batch):
    model_inputs, label_ids = batch
    outputs = self.model(**model_inputs, labels=label_ids)
    loss = outputs.loss
    logits = outputs.logits
    return loss, logits

  # This method defines what happens in a single training step. This calls the forward method
  # to do a forward pass

  #Arguments:
  # batch: a training batch
  # batch_idx: the index of the training batch

  #Returns: the loss returned by the forward method
  def training_step(self, batch, batch_idx):
    loss, logits = self.forward(batch)
    return {"loss": loss}

  #This method defines what happens in a single validation step  
  # Arguments: 
  # batch: a validation batch
  # batch_idx: the index of the validation batch

  #Returns: Whatever is returned by _evaludation_step method
  def validation_step(self, batch, batch_idx):
    return self._evaluation_step(batch) 

  #This method defines the functionality at the end of validation epoch(when all validation batches have been processed)
  #Arguments: 
  # outputs: The list of outputs returned by the validation_step method

  # Returns:
  # Whatever is returned by the _evaluation_epoch_end method
  def validation_epoch_end(self, outputs):
    self._evaluation_epoch_end(outputs, phase='test')

  #This method defines the functionality in a single test step

  # Arguments: 
  # batch: a single test batch
  # batch_idx: the index of the test step batch

  #Returns: Whatever is returned by the _evaluation_step_eval method 
  def test_step(self, batch, batch_idx):
    return self._evaluation_step_eval(batch) 

  #This method defines the functionality of the end of an epoch of test step

  # Arguments: 
  # outputs: The list of outputs from test_step method, where each element in the list
  #          corresponds to the output of a batch as returned by the test_step method

  #Returns: Whatever is returned by the _evaludation_epoch_end method
  def test_epoch_end(self, outputs):
    self._evaluation_epoch_end_eval(outputs, phase='test')

  #This method is called by the validation step method

  #Arguments: 
  # batch: a single batch of validation data

  # Returns: a dictionary containing the loss, the ground truth relations and the predicted relations

  def _evaluation_step(self, batch):
    loss, logits = self.forward(batch)
    _, label_ids = batch
    gts = [list(map(self.relations.__getitem__, label_id.nonzero(as_tuple=True)[0])) for label_id in label_ids]
    logits = torch.sigmoid(logits)
    pr_label_ids = torch.where(logits > 0.4, 1, 0)
    prs = [list(map(self.relations.__getitem__, label_id.nonzero(as_tuple=True)[0])) for label_id in pr_label_ids]

    return {
        "loss": loss.item(),
        "gts": gts,
        "prs": prs,
    }

  #This method is called by the validation_epoch_end method. 

  # Arguments:
  # outputs: The list of outputs, where each element is the output of a single evaluation batch
  def _evaluation_epoch_end(self, outputs, phase=None):
    ave_loss = np.mean([x["loss"] for x in outputs])
    gts = reduce(lambda x,y: x + y, [x["gts"] for x in outputs], [])
    prs = reduce(lambda x,y: x + y, [x["prs"] for x in outputs], [])
    acc = sum([list(set(gt).difference(set(['Unknown']))) == pr for gt, pr in zip(gts, prs)]) / len(gts)
    
    target_ids = random.sample(range(len(gts)), 5)
    
    print("="*50)
    print(f"ave_loss: {ave_loss}")
    print(f"ACC: {acc}")
    if phase == "test":
      print("GT" + "-"*40)
      for target_id in target_ids:
        print(f"GT: {gts[target_id]}\t\t\tPR: {prs[target_id]}")

  #This method is called by the test_step method 
  #Arguments: batch: a single batch of the test_set

  # Retuns a dictionary containing the logits of the relations in the knowledge graph(with key as "logits"),
  # the top 3 predicted relations(the key prs), the claim(the key sentences) 
  # and the entities in the claim(the key "entities") 
  def _evaluation_step_eval(self, batch):
    model_inputs, input_texts = batch
    outputs = self.model(**model_inputs)
    logits = outputs.logits
    pr_label_ids = torch.topk(logits, dim=-1, k=self.top_k).indices
    prs = [list(map(self.relations.__getitem__, pr_label_id)) for pr_label_id in pr_label_ids]
    sentences = [text.split('[sep]')[0] for text in input_texts]
    entities = [text.split('[sep]')[1] for text in input_texts]
    return {
        "logits": logits,
        "prs": prs,
        "sentences": sentences,
        "entities":entities
    }

  # This method is called by the test_epoch_end method. The method generates a pandas
  # dataframe with columns: claims, entities(the list of entities in the claim),
  # and the predicted relations. This dataframe is saved as a json file 
  # Arguments:
  # outputs: The list of outputs, where each element is the output of processing a test batch 
  def _evaluation_epoch_end_eval(self, outputs, phase=None):
    logits = [x["logits"] for x in outputs]
    prs = reduce(lambda x,y: x + y, [x["prs"] for x in outputs], [])
    #import pdb; pdb.set_trace()
    sentences = reduce(lambda x,y: x + y, [x["sentences"] for x in outputs], [])
    #input_texts = reduce(lambda x,y: x + y, [x["sentences"] for x in outputs], [])
    entities = reduce(lambda x,y: x + y, [x["entities"] for x in outputs], [])
    df = pd.DataFrame({'claims': sentences, 'entity': entities, 'output': prs})
    df.to_json(f"test_relations_top{self.top_k}.json")

  #This method initializes AdamW optimizer
  def configure_optimizers(self):
      grouped_params = [
          {
              "params": list(filter(lambda p: p.requires_grad, self.parameters())),
              "lr": self.learning_rate,
          },
      ]

      optimizer = torch.optim.AdamW(
          grouped_params,
          lr=self.learning_rate, 
      )
      return {"optimizer": optimizer}
