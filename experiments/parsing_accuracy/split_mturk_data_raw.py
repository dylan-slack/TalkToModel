"""Splits the raw data output from mturk into a test and validation set appropriate for the model eval and test sets.

For the diabetes data, there's a strange character that causes line breaks in the final data, but I can't seem to get
rid of it... After processing the diabetes data, I went through and removed these characters manually...
"""
import argparse
import json

import numpy as np


np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_location', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--acceptance_threshold', type=float, default=3.0)
args = parser.parse_args()

ACCEPTANCE_THRESHOLD = args.acceptance_threshold


with open(args.raw_data_location, 'r') as file:
    data = json.load(file)
    data_dict = json.loads(data)

original_sentence_hit_ids = list(data_dict.keys())
assert len(original_sentence_hit_ids) == 50, f"Original set has len {len(original_sentence_hit_ids)}!"
test_set_keys = original_sentence_hit_ids
assert len(test_set_keys) == 50, f"Test set has len {len(test_set_keys)}!"


def build_test_suite_string(hit_ids):
    return_string = ""
    for hid in hit_ids:
        hit = data_dict[hid]
        parse = hit["question_parse"]
        for revision_hit_id in hit["revisions"]:
            revision_scores = hit["revisions"][revision_hit_id]["scores"]
            revision_text = hit["revisions"][revision_hit_id]["revised_text"]
            scores = np.array(revision_scores)
            if np.mean(scores) >= ACCEPTANCE_THRESHOLD:
                if revision_text in return_string:
                    continue
                revision_text = revision_text.replace("\n", "")
                revision_text = revision_text.replace("\r", "")
                parse = parse.replace("\n", "")
                parse = parse.replace("\r", "")

                # resolve annotation errors
                parse = parse.replace("previous filter", "previousfilter")
                parse = parse.replace("score accuracy test", "score accuracy")
                parse = parse.replace("greater equal to", "greater equal than")

                return_string += revision_text + "\n"
                return_string += parse + "\n\n"
    while return_string.endswith("\n"):
        return_string = return_string[:-len("\n")]
    return return_string


def save(data_to_save, filename):
    with open(filename, 'w') as file_to_write:
        file_to_write.write(data_to_save)

test_string = build_test_suite_string(test_set_keys)
split_test_string = test_string.split('\n\n')
print(f"Test set has {len(split_test_string)} items")

test_file_name = f"{args.dataset}_test_suite.txt"
eval_file_name = f"{args.dataset}_eval_suite.txt"

save(test_string, test_file_name)