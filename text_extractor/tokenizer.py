import logging
from typing import List, Union

import pandas as pd
import torch

from .simple_tokenizer import SimpleTokenizer as _Tokenizer


class Tokenizer:
    def __init__(self, nc, nl, td='./sem_info'):
        self._tokenizer = _Tokenizer()
        self.t = {k: pd.read_csv(f'{td}/{k}_{nc}.csv').values.tolist() for k in nl}
        
    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def gen_token(self):
        token_dict = {k: torch.cat([self.tokenize(i[1]) for i in self.t[k]]) for k in self.t.keys()}
        
        return token_dict
    