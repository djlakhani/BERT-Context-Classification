import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import ScenarioModel, SupConModel, CustomModel, SupConClassifierHead
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, eps = args.adam_epsilon)
    
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            losses += loss.item()
    
        run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()
    # 1: setup train dataloader

    # 2: setup model's optimizer_scheduler if you have
      
    # 3: write a training loop

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        # batch is a tuple with 5 elements and each element contains 16 (batch_size) objects
        # The first is the input tokens,
        # second is the input_ids, third is attention mask
        # fourth is input labels, and finally text labels.
        inputs, labels = prepare_inputs(batch)
        logits = model(inputs, labels)
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss()
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, eps = args.adam_epsilon)

    # first training the encoder: BERT representation
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch)
            features1 = model(inputs, None)
            features2 = model(inputs, None)
            features = torch.stack([features1, features2], dim=1)
            
            loss = criterion(features, labels, mask=None)

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            losses += loss.item()
    
        # run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)

    
    
    # training the decoder: SupConModel classifier head

    target_size=18
    hidden_dim =768

    # freeze weights of BERT model
    for param in model.parameters():
        param.requires_grad = False
        
    classifier_model = SupConClassifierHead(hidden_dim, target_size).to(device)
    criterion_classify = nn.CrossEntropyLoss()
    optimizer_classify = torch.optim.Adam(classifier_model.parameters(), lr = 1e-2) #, eps = args.adam_epsilon) #1e-3
    dataloader = get_dataloader(args, datasets['train'], split='train')

    accuracy = []
    train_accuracy = []
    epochs = []

    for epoch_count in range(args.n_epochs):       
        losses = 0
        classifier_model.train()

        for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
            inputs, labels = prepare_inputs(batch, use_text=False)
            model_out = model(inputs, None)
            logits = classifier_model(model_out)
            loss = criterion_classify(logits, labels)
            
            loss.backward()
            optimizer_classify.step()  # backprop to update the weights
            classifier_model.zero_grad()
            losses += loss.item()

        # train_acc = run_eval(args, classifier_model, datasets, tokenizer, split='train')
        # acc = run_eval(args, classifier_model, datasets, tokenizer, split='validation')
        # accuracy.append(acc * 100)
        # train_accuracy.append(train_acc * 100)
        # epochs.append(epoch_count)
        
        print('epoch', epoch_count, '| losses:', losses)



    # compute test accuracy
    test_dataloader = get_dataloader(args, datasets["test"], "test")
    acc = 0
    classifier_model.eval()
    model.eval()
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch)
        model_out = model(inputs, None)
        logits = classifier_model(model_out)
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'test acc:', acc/len(datasets['test']))

if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
 
  if args.task == 'baseline':
    model = ScenarioModel(args, tokenizer, target_size=18).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    model = CustomModel(args, tokenizer, target_size=18).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=18).to(device)
    supcon_train(args, model, datasets, tokenizer)
   
