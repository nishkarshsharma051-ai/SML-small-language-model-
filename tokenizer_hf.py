from transformers import PreTrainedTokenizer
from typing import List, Optional

class TingLingLingTokenizer(PreTrainedTokenizer):
    def __init__(self, chars=None, model_max_length=1024, **kwargs):
        self.chars = chars if chars else []
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
        super().__init__(
            model_max_length=model_max_length,
            **kwargs
        )
        
        # Add special tokens
        if not self.chars:
            # Fallback for empty init
            self.chars = ["<unk>", "<s>", "</s>", "<pad>"]
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def get_vocab(self):
        return self.stoi

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.stoi.get(token, self.stoi.get("<unk>", 0))

    def _convert_id_to_token(self, index):
        return self.itos.get(index, "<unk>")

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        import os
        import json
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.chars, f, ensure_ascii=False)
        
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        import os
        import json
        if os.path.isdir(pretrained_model_name_or_path):
            vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
            if os.path.exists(vocab_file):
                with open(vocab_file, "r") as f:
                    chars = json.load(f)
                return cls(chars=chars, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
