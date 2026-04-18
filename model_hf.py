import torch
from transformers import PreTrainedModel, PretrainedConfig
from model import TingLingLing, TingLingLingConfig

class TingLingLingHFConfig(PretrainedConfig):
    model_type = "ting_ling_ling"
    def __init__(
        self,
        vocab_size=65,
        n_embd=384,
        n_layer=6,
        n_head=6,
        block_size=256,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        super().__init__(**kwargs)

class TingLingLingModelHF(PreTrainedModel):
    config_class = TingLingLingHFConfig
    
    def __init__(self, config):
        super().__init__(config)
        # map HF config to our custom model config
        custom_config = TingLingLingConfig(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            block_size=config.block_size
        )
        self.model = TingLingLing(custom_config)
        
    def forward(self, input_ids, labels=None, **kwargs):
        logits, loss = self.model(input_ids, labels)
        
        # HF expects a CausalLMOutput-like object or a tuple
        return (loss, logits) if loss is not None else logits

    def generate(self, input_ids, **kwargs):
        # Delegate to the underlying custom model's generation logic
        return self.model.generate(input_ids, **kwargs)
