
import json
import time
import numpy as np


class Validator:

    def __init__(self, llm):
        from llm import LLM 
        self.llm: LLM = llm
        self.buffer = ""
        self.generated = []

        self.allowed_chars = set('{}[]":,0123456789.truefalsn- ')
        self.structural_tokens = None
        self.all_tokens = None

        self.timings = {
            "vocab_decode": 0.0,
            "inside_string": 0.0,
            "advance_decode": 0.0,
            "json_loads": 0.0
        }

    def initialize_vocab(self, vocab_size):
        self.all_tokens = np.arange(vocab_size, dtype=np.int64)

        structural = []

        start = time.perf_counter()

        for token_id in range(vocab_size):
            text = self.llm.decode([token_id])

            if text and all(char in self.allowed_chars for char in text):
                structural.append(token_id)

        self.timings["vocab_decode"] += time.perf_counter() - start

        self.structural_tokens = np.array(
            structural,
            dtype=np.int64
        )

    def initial_state(self):
        self.buffer = ""
        self.generated = []

        for key in self.timings:
            self.timings[key] = 0.0

        return self

    @property
    def inside_string(self):
        start = time.perf_counter()

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

        self.timings["inside_string"] += time.perf_counter() - start

        return inside

    def allowed_tokens(self):
        if self.inside_string:
            return self.all_tokens

        return self.structural_tokens

    def advance(self, token_id):
        self.generated.append(token_id)

        start = time.perf_counter()
        self.buffer = self.llm.decode(self.generated)
        self.timings["advance_decode"] += time.perf_counter() - start

        return self

    @property
    def is_accepting(self):
        start = time.perf_counter()

        try:
            result = isinstance(json.loads(self.buffer), dict)
        except json.JSONDecodeError:
            result = False

        self.timings["json_loads"] += time.perf_counter() - start

        return result
