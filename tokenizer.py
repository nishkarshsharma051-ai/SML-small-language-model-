class CharacterTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({'chars': self.chars}, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        instance = cls("")
        instance.chars = data['chars']
        instance.vocab_size = len(instance.chars)
        instance.stoi = { ch:i for i,ch in enumerate(instance.chars) }
        instance.itos = { i:ch for i,ch in enumerate(instance.chars) }
        return instance
