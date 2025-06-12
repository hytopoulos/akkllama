
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle

class EvaCun(Dataset):
    '''
    EvaCun Akkadian Cuneiform dataset
    '''

    def __init__(self, path, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.data = self.preprocess(path)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_ids = item['input_ids'].clone()
        attention_mask = item['attention_mask'].clone()
        
        # choose 15% of non-padding tokens to predict
        prob_matrix = torch.full(input_ids.shape, 0.15, device=input_ids.device)
        prob_matrix = prob_matrix * attention_mask
        masked_pos = torch.bernoulli(prob_matrix).bool()
        
        # build labels: only keep masked positions, ignore the rest
        labels = input_ids.clone()
        labels[~masked_pos] = -100
        
        # apply 80/10/10 corruption on the masked positions
        # sample a uniform random in [0,1) for every position
        probs = torch.rand(input_ids.shape, device=input_ids.device)
        
        # 80% [MASK]
        mask_mask   = masked_pos & (probs < 0.8)
        input_ids[mask_mask] = self.tokenizer.pad_token_id

        # 10% random replacement
        rand_mask = masked_pos & (probs >= 0.8) & (probs < 0.9)        
        random_tokens = torch.randint(
            low=0,
            high=self.tokenizer.vocab_size,
            size=input_ids.shape,
            device=input_ids.device,
        )
        input_ids[rand_mask] = random_tokens[rand_mask]
        
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels
        }

    def __len__(self):
        return len(self.data)

    def preprocess(self, path):
        '''
        Preprocess EvaCun csv file into tokenized sequences
        '''

        df = pd.read_csv(path)
        df['value'] = df['value'].astype(str)

        df['fragment_line_num'] = pd.to_numeric(df['fragment_line_num'], errors='coerce')
        df = df.dropna(subset=['fragment_line_num'])
        df['fragment_line_num'] = df['fragment_line_num'].astype(int)

        df['index_in_line'] = pd.to_numeric(df['index_in_line'], errors='coerce')
        df = df.dropna(subset=['index_in_line'])
        df['index_in_line'] = df['index_in_line'].astype(int)

        # some words are missing from the dataset (indicative of damage on the original tablet),
        # we need to fill with pad tokens
        documents = []
        for frag_id, frag_df in df.groupby('fragment_id'):
            tokens = []
            # now fragment_line_num is all ints, so this will sort
            for line_num, line_df in (frag_df.sort_values('fragment_line_num').groupby('fragment_line_num')):
                max_idx   = line_df['index_in_line'].max()
                token_map = dict(zip(line_df['index_in_line'], line_df['value']))
                for idx in range(1, max_idx + 1):
                    tokens.append(token_map.get(idx, self.tokenizer.pad_token))
            documents.append("".join(tokens))

        # tokenize + chunking
        block_size = 512
        text_tokenized = self.tokenizer(documents, pad_to_multiple_of=block_size, return_attention_mask=True)
        ds = []
        for input_ids, attention_mask in zip(text_tokenized.input_ids, text_tokenized.attention_mask):
            for i in range(0, len(input_ids), block_size):
                chunk_input = input_ids[i : i + block_size]
                chunk_attention = attention_mask[i : i + block_size]

                pad_len = block_size - len(chunk_input)
                if pad_len > 0:
                    if self.tokenizer.padding_side == "left":
                        chunk_input = [self.tokenizer.pad_token_id] * pad_len + chunk_input
                        chunk_attention = [0] * pad_len + chunk_attention
                    else:
                        chunk_input = chunk_input + [self.tokenizer.pad_token_id] * pad_len
                        chunk_attention = chunk_attention + [0] * pad_len

                # ensure there is no attention on pad tokens
                ids  = torch.tensor(chunk_input, dtype=torch.long)
                mask = torch.tensor(chunk_attention, dtype=torch.long)
                mask = mask.masked_fill(ids == self.tokenizer.pad_token_id, 0)

                ds.append({"input_ids": ids, "attention_mask": mask})

        self.df = df
        return ds

    def save(self, path):
        self.df.to_csv(path, index=False)

    def load(self, path):
        self.df = pd.read_csv(path)
        self.data = self.preprocess(path)
