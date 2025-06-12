import os, pickle
import torch
import wandb
from collections import defaultdict
import os, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
from models import MNTPModel
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from dataset import EvaCun

class Agent:
    def __init__(self, args, load_path=""):
        self.args = args
        self.device = args.device if args.device != 'auto' else 'cuda'

        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.load_or_resume(args)

    def save_checkpoint(self, args, path):
        ''' Save the adapter, optimizer, scheduler, dataset and training stats '''
        if not os.path.exists(path):
            os.makedirs(path)

        self.model.save(path)
        self.train_dataset.save(f"{path}/akk_train.csv")
        self.val_dataset.save(f"{path}/akk_val.csv")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pt")
        pickle.dump(self.train_metrics, open(f"{path}/train_metrics.pkl", "wb"))
        pickle.dump(self.val_metrics, open(f"{path}/val_metrics.pkl", "wb"))
        pickle.dump(self.args, open(f"{path}/args.pkl", "wb"))

    def load_or_resume(self, args):
        ''' Initialize tokenizer and base model.
            Load the adapter, optimizer, scheduler, dataset and training stats '''

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
        except Exception as e:
            print(f"Could not load tokenizer from {args.model}, defaulting to codebert-base")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", use_fast=True, trust_remote_code=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.train_dataset = EvaCun(os.path.join(args.load, 'akk_train.csv'), self.tokenizer, args)
        self.val_dataset = EvaCun(os.path.join(args.load, 'akk_val.csv'), self.tokenizer, args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        self.model = MNTPModel(
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            args=args,
        )

        self.optimizer = torch.optim.AdamW([{
            'params': list(self.model.model.parameters()),
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }])

        num_training_steps = len(self.train_dataset) * self.args.epoch
        match args.scheduler:
            case 'linear':
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=num_training_steps,
                )
            case 'constant':
                self.scheduler = get_constant_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                )
            case 'cosine':
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=num_training_steps,
                )
            case 'cosine_with_restarts':
                self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=num_training_steps,
                )
            case _:
                raise ValueError(f"Unknown scheduler: {args.scheduler}")

        if args.load:
            self.optimizer.load_state_dict(torch.load(f"{args.load}/optimizer.pt"))
            self.scheduler.load_state_dict(torch.load(f"{args.load}/scheduler.pt"))
            self.train_metrics = pickle.load(open(f"{args.load}/train_metrics.pkl", "rb"))
            self.val_metrics = pickle.load(open(f"{args.load}/val_metrics.pkl", "rb"))

    def train(self):
        ''' Train the model '''

        if not self.args.load:
            self.train_metrics['epoch'] = 0
            self.train_metrics['step'] = 0
            self.val_metrics['step'] = 0

        wandb.watch(self.model, log='all', log_freq=self.args.log_steps)

        for epoch in range(self.train_metrics['epoch'], self.args.epoch):
            self.train_metrics['epoch'] = epoch
            self.train_one_epoch()

            if self.train_metrics['epoch'] % self.args.val_interval == 0:
                self.validate()

            if not self.args.disable_checkpoint and self.train_metrics['epoch'] == self.args.epoch - 1:
                self.save_checkpoint(self.args, f'{wandb.run.dir}/final')
                print(f'Final model saved to {wandb.run.dir}/final')

    def train_one_epoch(self):
        ''' Train the model for one epoch '''
        
        self.model.train()

        losses, accs = [], []

        tqdm_batch = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Train [Epoch {self.train_metrics["epoch"]}]')
        for i, batch in enumerate(tqdm_batch):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()

            logits = self.model.forward(batch)

            if self.args.model.find('bert') == -1:
                input_ids = input_ids[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()

            preds = logits.argmax(dim=-1)
            acc = (preds[labels != -100] == labels[labels != -100]).sum().item() / (labels != -100).sum().clamp(min=1).item()
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # check if loss is nan
            if torch.isnan(loss):
                print("Loss is NaN, skipping batch")
                continue

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            losses += [loss.item()]
            accs += [acc]
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})

            # logging
            self.train_metrics['step'] += 1
            if (self.train_metrics['step'] % self.args.log_steps == 0) or (i == len(self.train_loader) - 1):
                self.train_metrics['loss'].append(np.mean(losses[-self.args.log_steps:]))
                self.train_metrics['acc'].append(np.mean(accs[-self.args.log_steps:]))
                wandb.log({
                    'train_loss': self.train_metrics['loss'][-1],
                    'train_acc': self.train_metrics['acc'][-1],
                    'train_step': self.train_metrics['step'],
                })
 
            if self.train_metrics['step'] % self.args.val_steps == 0:
                self.validate()

    @torch.no_grad()
    def validate(self):
        ''' Compute accuracy and loss on the validation set '''
        
        self.model.eval()

        losses = []

        all_labels = []
        all_preds = []

        tqdm_batch = tqdm(self.val_loader, total=len(self.val_loader), desc='Val')
        for i, batch in enumerate(tqdm_batch):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model.forward(batch)

            if self.args.model.find('bert') == -1:
                input_ids = input_ids[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()

            preds = logits.argmax(dim=-1) # + self.model.model.model.config.vocab_size - self.num_trainable_tokens
            all_labels.append(labels[labels != -100].cpu().numpy())
            all_preds.append(preds[labels != -100].cpu().numpy())
            # acc = (preds[labels != -100] == input_ids[labels != -100]).sum().item() / (labels != -100).sum().clamp(min=1).item()
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # check if loss is nan
            if torch.isnan(loss):
                print("Loss is NaN, skipping batch")
                continue

            losses += [loss.item()]
            tqdm_batch.set_postfix({'loss': np.mean(losses)})

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        acc = (all_preds == all_labels).sum() / len(all_labels)

        # logging
        self.val_metrics['step'] += 1
        self.val_metrics['loss'].append(sum(losses)/len(losses))
        self.val_metrics['acc'].append(acc)
        wandb.log({
            'val_loss': self.val_metrics['loss'][-1],
            'val_acc': self.val_metrics['acc'][-1],
            'val_step': self.val_metrics['step'],
        })
        if not self.args.disable_checkpoint and min(self.val_metrics['loss']) == self.val_metrics['loss'][-1]:
            self.save_checkpoint(self.args, f'{wandb.run.dir}/best')
            print(f'Best model saved to {wandb.run.dir}/best')

    @torch.no_grad()
    def test(self, test_loader):
        ''' Evaluate the model on a test set; compute accuracy and MRR (Mean Reciprocal Rank) '''
        
        self.model.eval()

        losses = []
        all_labels = []
        all_preds = []
        reciprocal_ranks = []

        tqdm_batch = tqdm(test_loader, total=len(test_loader), desc='Test')
        for i, batch in enumerate(tqdm_batch):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model.forward(batch)
            if self.args.model.find('bert') == -1:
                input_ids = input_ids[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()

            preds = logits.argmax(dim=-1)

            # Flatten logits and labels to compute loss and MRR on token level
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            # Filter out padding tokens
            valid_indices = labels_flat != -100
            logits_valid = logits_flat[valid_indices]
            labels_valid = labels_flat[valid_indices]

            # Compute loss
            loss = self.loss_fn(logits_valid, labels_valid)
            if torch.isnan(loss):
                print("Loss is NaN, skipping batch")
                continue

            losses += [loss.item()]

            # Compute accuracy
            preds_flat = preds.view(-1)[valid_indices]
            all_labels.append(labels_valid.cpu().numpy())
            all_preds.append(preds_flat.cpu().numpy())

            # Compute MRR
            ranks = torch.argsort(torch.argsort(-logits_valid, dim=1), dim=1) + 1
            label_ranks = ranks[torch.arange(len(labels_valid)), labels_valid]
            rr = (1.0 / label_ranks.float()).cpu().numpy()
            reciprocal_ranks.extend(rr)

            tqdm_batch.set_postfix({'loss': np.mean(losses)})

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        acc = (all_preds == all_labels).sum() / len(all_labels)
        mrr = np.mean(reciprocal_ranks)

        return acc, mrr
