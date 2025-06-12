import torch
import torch.nn as nn
from llm2vec.llm2vec.models import LlamaBiModel, LlamaBiForMNTP, GemmaBiForMNTP
import os

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    TrainableTokensConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

class FeatureExtractor(nn.Module):
    '''
    Base class for feature extractors
    '''

    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs['args']
        self.tokenizer = kwargs['tokenizer']
        self.device = self.args.device if self.args.device != 'auto' else 'cuda'

    def load(self, path):
        raise NotImplementedError()
    
    def save(self, path):
        raise NotImplementedError()

    def forward(self, batch):
        raise NotImplementedError()

class EmbeddingModel(FeatureExtractor):
    '''
    Base class for embedding models
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # quantization options
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["classifier", "pre_classifier"] if self.args.model.find('bert') != -1 else None,
        )

        cl = AutoModelForCausalLM if self.args.model.find('bert') != -1 else LlamaBiForMNTP

        # load pretrained model
        self.model = cl.from_pretrained(
            self.args.model,
            device_map=self.args.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            revision="main",
            # num_labels=len(args.classes),
            quantization_config=quant_config if self.args.model.find('bert') == -1 else None,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

class LoRAModel(EmbeddingModel):
    '''
    Model fine-tuned using LoRA
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        lora_config = LoraConfig(
                r = self.args.lora_r,
                lora_alpha = self.args.lora_alpha,
                lora_dropout = 0.1,
                target_modules='all-linear',
                task_type="FEATURE_EXTRACTION" if self.args.encoder else "CAUSAL_LM",
                inference_mode=False,
        )

        self.model = prepare_model_for_kbit_training(
            self.model,
            # TODO: Do all models that do not support auto also not support gradient checkpointing?
            use_gradient_checkpointing=(self.args.device == 'auto'),
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        self.model = get_peft_model(self.model, lora_config)

    def load(self, path):
        set_peft_model_state_dict(self.model, torch.load(f'{path}/adapter_model.pt'))
        self.model = self.model.to(self.model.device)
    
    def save(self, path):
        torch.save(get_peft_model_state_dict(self.model), f'{path}/adapter_model.pt')

class LayerFTModel(EmbeddingModel):
    '''
    Model fine-tuned using layer-wise unfreezing
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_frozen(self.args.ft_layers)

    def save(self, path):
        torch.save(self.model.state_dict(), f'{path}/model.pt')

    def load(self, path):
        self.model.load_state_dict(torch.load(f'{path}/model.pt'))
        self.model = self.model.to(self.model.device)
        
        self.set_frozen(self.args.ft_layers)

    def set_frozen(self, num_layers, from_top=True):
        '''
        Freeze all parameters except the last `num_layers` layers
        '''

        for param in self.model.parameters():
            param.requires_grad = False

        if num_layers == 0:
            return

        if from_top:
            for param in self.model.roberta.encoder.layer[-self.args.ft_layers:].parameters():
                param.requires_grad = True
        else:
            for param in self.model.roberta.encoder.layer[:self.args.ft_layers].parameters():
                param.requires_grad = True

class MNTPModel(nn.Module):
    '''
    Model for masked next token prediction (MNTP)
    '''
    
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs['args']
        self.device = self.args.device if self.args.device != 'auto' else 'cuda'
        self.model = LoRAModel(**kwargs) if not self.args.ft_layers else LayerFTModel(**kwargs)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
        self.model = self.model.to(self.device)

    def forward(self, batch):
        return self.model(batch)
