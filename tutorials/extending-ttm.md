# Extending TalkToModel

It's fairly straightforward to build on the TTM system to implement new functionality. In this tutorial, we walk through how to implement a new system capability.

## Overview

Here we discuss a bit about the repository is organized and how the system executes different capabilities.  At a high level, TalkToModel parses natural language utterances in parses in a grammar.

A natural language utterance might look like:

- *"What do you predict on men in the data?"*

and the corresponding parse is

- *filter male true and predict*

Each *operation* in the grammar (*filter* and *predict* in this case) accepts some arguments, like the feature in the dataset male and the value to filter on (*true*).

Consequently, implementing a **new** system capability involves:

1. Updating the grammar to include the new operation and arguments
2. Implementing prompts for the new operation, so the system learns to parse it
3. Writing the code that executes the operation when it is called during parsing. This code can do things like modify the state of the conversation, the dataset during the parse, and append text to the output that's ultimately returned to the user.

## Extending The Grammar

The grammar is stored in `./explain/grammar.py`. It's written in [Lark](https://lark-parser.readthedocs.io/en/latest/grammar.html).

To see what updating the grammar for a new operation would look like, let's imagine we want to add a new operation that computes the max of a feature. We would add the operation
```shell
maxfeature: " featuremax" allfeaturenames
```
as a new line in the file. Here, " featuremax" is the name of the operation and allfeaturenames are the acceptable arguments. In this case, this all the feature names in the dataset, which is already defined by the allfeaturenames non-terminal. Also, we would add it to the existing operations via appending `| maxfeature` to the line beginning with `operation:` as follows
```shell
operation: ... | maxfeature
```
That's all the updates to the grammar that are needed! The system can now parse operations like "What's the max age of men in the data" ==> `filter male true and featuremax age`

## Adding Prompts

To add prompts for the new operation, you can create a new prompt file `featuremax.txt` in `./explain/prompts/dynamic/featuremax.txt`. This file will store the prompts used to train the system to parse natural language into this operation correctly.

For example, you can write
```txt
User: for {filter_text}, what's the max {num_features}?
Parsed: {filter_parse} and featuremax {num_features} [E]
```
This prompt will substitute filtering text and the numeric feature names into `{filter_text}` and `{num_features}` respectively to create a larger prompt set. For instance, it might generate
```txt
User: for single men, what's the max income?
Parsed: filter single true and male true and featuremax income [E]

User: for married men, what's the max loan amount?
Parsed: filter married true and male true and featuremax loan amount [E]
```
In general, it's useful to write as many prompts as possible. There's many examples of prompts in the `./explain/prompts/dynamic` directory. 

## Writing Execution Code

Last, we need to implement the code that is called when the operation gets executed. For example, when our `featuremax` operation gets called.

To do this, we will add a new file in the `./explain/actions` directory called `max.py`. Here's our function for computing the max. The `actions` directory contains code for each of the operations supported by the system.

```python
def max_operation(conversation, parse_text, i, **kwargs):
    """Generates the max along a feature."""
    # get the feature name
    feature_name = parse_text[i+1]
    # get the max value
    max_value = conversation.temp_dataset.contents['X'][feature_name].max()
    # compose the return string
    return_string = f"The max value of {feature_name} is {max_value}."
    # return the string and 1, indicating success
    return return_string, 1
```

The operation functions have a common signature, where conversation is the current state of the conversation, parse_text is the parsed text, i is an index in the parse, and **kwargs are extra keyword arguments. 

To hook this operation into the conversation, we edit `./explain/actions/get_action_functions.py`. This file contains a dictionary called `actions`. We add
```python
actions = {
    ...
    "featuremax": max_operation
}
```
to the dictionary and now when `featuremax` is parsed by the system, out operation will be called! We should be good to go at this point.