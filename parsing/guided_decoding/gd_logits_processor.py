"""Guided Decoding logits processor"""
import copy
import re
from tqdm import tqdm

from lark import Lark
import numpy as np

import torch

from transformers import LogitsProcessor


class GuidedDecodingLogitsProcessor(LogitsProcessor):
    def __init__(self, parser, prompt_length, filter_value=-float("Inf"), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = parser
        self.prompt_length = prompt_length
        self.filter_value = filter_value

    def __call__(self, input_ids, scores):
        valid_tokens = torch.ones_like(scores) * self.filter_value

        # The tokens generated so far
        for b in range(scores.shape[0]):
            generated_tokens = input_ids[b, self.prompt_length:].cpu().tolist()
            next_tokens = self.parser.next_tokens(generated_tokens)
            int_next_tokens = np.array([int(t) for t in next_tokens])

            # Adjust the scores to allow only valid tokens
            valid_tokens[b, int_next_tokens] = scores[b, int_next_tokens]
        return valid_tokens


class GuidedParser:
    """A class defining the mapping between text grammar and tokenized grammar."""
    def __init__(self, init_grammar, tokenizer, model, eos_token=None):

        # The grammar with natural language text
        self.text_grammar = init_grammar

        # The natural language parser
        self.text_parser = Lark(self.text_grammar, parser="lalr")

        # The hugging face tokenizer
        self.tokenizer = tokenizer

        # Store the model being used. This influences some decoding settings
        self.model = model

        # The grammar compiled with tokens from the hugging face tokenizer
        self.token_grammar = self._compile_grammar(self.text_grammar, self.tokenizer)

        # The tokenized parser
        self.token_parser = Lark(self.token_grammar, parser="lalr")

        self.terminal_lookup = {}

        for terminal in self.token_parser.terminals:
            self.terminal_lookup[terminal.name] = terminal.pattern.value

        if eos_token is None:
            if model == "t5":
                self.eos_token = tokenizer.encode(" [e]")[-2]
            elif model == "gpt":
                self.eos_token = tokenizer.encode(" [e]")[-1]
            else:
                raise NameError(f"don't know model {model}")
        else:
            self.eos_token = eos_token

    def _compile_grammar(self, grammar, tokenizer):
        """Compiles a grammar into tokens."""

        # Create the tokenizer grammar
        tokenized_grammar = copy.deepcopy(grammar)

        # Find all the terminals
        terminals = re.findall('"([^"]*)"', grammar)

        # Store existing terminals
        existing_terms = {}

        # Records the update rules for the terminals
        indx = 0
        for term in tqdm(terminals):
            tokens = tokenizer.encode(term)

            replacement_rule = "("
            for tok in tokens:
                if tok == 1 and self.model == "t5":
                    continue
                # If it already exists, we don't want to add
                # the terminal again, just use the old one
                if tok in existing_terms:
                    name = existing_terms[tok]
                else:
                    name = f"ANON{indx} "
                    indx += 1
                    newrule = name + ": " + "\"" + str(tok) + "\""
                    tokenized_grammar += f"\n{newrule}"
                    existing_terms[tok] = name
                replacement_rule += name

            # Close the list of terminals
            replacement_rule += ")"

            # Update the terminal with the tokens
            tokenized_grammar = tokenized_grammar.replace("\"" + term + "\"",  replacement_rule)

        tokenized_grammar += "\n%ignore \" \""
        return tokenized_grammar

    def next_tokens(self, tokens):
        """Get the next tokens."""
        string_tokens = ' '.join([str(t) for t in tokens])
        interactive = self.token_parser.parse_interactive(string_tokens)
        interactive.exhaust_lexer()
        return [self.terminal_lookup[acc] for acc in interactive.accepts()]
