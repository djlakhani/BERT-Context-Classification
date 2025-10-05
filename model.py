import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig

class ScenarioModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size
    self.args = args
    self.dropout = nn.Dropout(p=args.drop_rate)
    self.classify = Classifier(args, target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")
    self.encoder = BertModel.from_pretrained("bert-base-uncased")
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  def forward(self, inputs, targets):
    outputs = self.encoder(**inputs)
    last_hidden_cls = outputs.last_hidden_state[:, 0, :]
    cls_dropout = self.dropout(last_hidden_cls)

    return self.classify(cls_dropout)

  
class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)
    
    # use initialization for setting different strategies/techniques to better fine-tune the BERT model

class SupConModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # initialize a linear head layer
    self.linear_supcon = nn.Linear(feat_dim, feat_dim)
 
  def forward(self, inputs, targets):

    """
    1: 
        feeding the input to the encoder, 
    2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    features = self.encoder(**inputs)
    features_dropout = self.dropout(features.last_hidden_state[:, 0, :])
    features_normalized = torch.nn.functional.normalize(features_dropout, dim=1)
    return self.linear_supcon(features_normalized)


class SupConClassifierHead(nn.Module):
  def __init__(self, hidden_dim=768, target_size=18):
    super().__init__()
    self.classifier = nn.Linear(hidden_dim, target_size)
    
  def forward(self, inputs, targets=None):
    return self.classifier(inputs)
