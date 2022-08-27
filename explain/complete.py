"""Completion engine for ava machines.

Note that the ava servers have be serving a completion model
for this code to work.
"""
from typing import Any

import gin
import requests


@gin.configurable
class AvaCompletion:
    """Perform a completion using an ava machine.

    Note, that for this to work, `serve.py` must be running on
    the requested machine.
    """
    def __init__(self, url='https://nlp.ics.uci.edu/gpt/'):
        self.url = url

    # The location of the API
    def create(self,
               prompt: str,
               num_sequences: int = 1,
               max_length: int = 30,
               temperature: float = 0.4) -> Any:
        """Create a completion."""
        query = self.url + '/complete/'

        params = {
            'max_length': max_length,
            'temperature': temperature,
            'num_sequences': num_sequences,
            'prompt': prompt,
        }

        resp = requests.post(url=query, json=params)
        data = resp.json()

        return data

    def guided_create(self,
                      prompt: str,
                      grammar: str,
                      max_len: int = 100) -> Any:
        """Perform a guided completion."""
        query = self.url + '/guided-completion/'
        params = {
            'prompt': prompt,
            'grammar': grammar,
            'max_len': max_len
        }

        resp = requests.post(url=query, json=params)
        data = resp.json()
        return data
