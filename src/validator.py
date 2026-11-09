import json
import numpy as np

class Validator:

    def __init__(self, llm):
        from src.llm import LLM
        self.llm: LLM = llm
        self.buffer = ""
        self.generated = []

        self.allowed_chars = set('{}[]":,0123456789.truefalsn- ')
        self.structural_tokens = None
        self.all_tokens = None
    
    def initialize_vocab(self, vocab_size):
        self.all_tokens = np.arange(vocab_size, dtype=np.int64)

        structural = []

        for token_id in range(vocab_size):
            text = self.llm.decode([token_id])

            if text and all(char in self.allowed_chars for char in text):
                structural.append(token_id)

        self.structural_tokens = np.array(
            structural,
            dtype=np.int64
        )

    def initial_state(self):
        self.buffer = ""
        self.generated = []
        return self

    @property
    def inside_string(self):
        escaped = False
        inside = False

        for char in self.buffer:
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                inside = not inside

        return inside

    def allowed_tokens(self):
        if self.inside_string:
            return self.all_tokens

        return self.structural_tokens

    def advance(self, token_id):
        self.generated.append(token_id)
        self.buffer = self.llm.decode(self.generated)
        return self

    @property
    def is_accepting(self):
        try:
            return isinstance(json.loads(self.buffer), dict)
        except json.JSONDecodeError:
            return False