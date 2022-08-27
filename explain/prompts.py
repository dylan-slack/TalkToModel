"""Generates prompts, embeds them, and finds nearest neighbors.

This class controls the prompt generation, embedding, and finding
nearest neighbors.
"""
import copy
import os
import string
import warnings
from random import shuffle
from typing import Union

import gin
import numpy as np
import pickle as pkl

import wordninja
from tqdm import tqdm

from flask import Flask
from sentence_transformers import SentenceTransformer, util
import torch
from word2number import w2n

from explain.utils import find_csv_filenames

app = Flask(__name__)


def get_user_part_of_prompt(prompt: str):
    """Scrapes the user part of a prompt.

    Arguments:
        prompt: The prompt
    Returns:
        user_component: The user input component of prompt
    """
    user_part = prompt.split("\n")[0]
    user_component = user_part[len("user: "):]
    return user_component


def build_semantic_dict(cat_filler_dict, seed: int = 0):
    semantic_cat_names = {
        f: ' '.join(wordninja.split(f)) for f in cat_filler_dict
    }
    return semantic_cat_names


@gin.configurable
class Prompts:
    def __init__(self,
                 prompt_folder: str,
                 cat_features: list[str],
                 num_features: list[str],
                 feature_value_dict: dict,
                 target: Union[list[int], list[str]],
                 class_names: dict,
                 skip_creating_prompts: bool = False,
                 num_per_knn_prompt_template: int = 1,
                 num_prompt_template: int = 15,
                 prompt_cache_size: int = 1_000_000,
                 prompt_cache_location: str = './cache/prompts.pkl',
                 max_values_per_feature: int = 50,
                 sentence_transformer_model_name: str = 'all-mpnet-base-v2',
                 filter_filler_dict_loc: str = "./explain/prompts/filter_fillers.txt"):
        """Init. This routine generates the prompts and embeds them.

        Arguments:
            prompt_folder: The folder where the prompts are stored.
            cat_features: The names of the categorical features.
            num_features: The names of the numeric features.
            feature_value_dict: A dictionary mapping from feature names to possible values.
            target: an array containing the model outputs used to generate prompts
            class_names: dictionary mapping between class values and names
            skip_creating_prompts: whether to skip prompt creation
            num_per_knn_prompt_template: Max Number of prompts to use from each prompt template
            num_prompt_template: Number of prompt templates to use.
            prompt_cache_size: Max allowable size of the prompt cache.
            prompt_cache_location: The location of the prompt cache.
            max_values_per_feature: The number of values to sample for generating prompts
                                    from features.
            sentence_transformer_model_name: The name of the embedding model.
        """
        self.final_prompt_set = None
        self.filename_to_prompt_id = None
        self.prompt_folder = prompt_folder

        self.prompt_cache_size = prompt_cache_size
        self.prompt_cache_location = prompt_cache_location
        self.prompt_cache = self._load_prompt_cache(prompt_cache_location)
        self.num_per_knn_prompt_template = num_per_knn_prompt_template
        self.num_prompt_template = num_prompt_template

        self.cat_features = cat_features
        self.num_features = num_features

        self.filter_filler_dict_loc = filter_filler_dict_loc

        # Sentence embedding
        # Consider pushing embedding model to server and call through API
        self.sentence_emb_model = SentenceTransformer(
            sentence_transformer_model_name)

        # Note(dylan): Is there a way to look up the embedding dim? this
        # is different between models, and didn't see a way to do it when
        # I looked at it for a moment
        self.embedding_dim = self.sentence_emb_model.encode(
            "whoopdadooo", show_progress_bar=False).shape[0]

        self.skip = skip_creating_prompts

        if not skip_creating_prompts:
            self.generate_prompts(cat_features, num_features, target, class_names,
                                  feature_value_dict, max_values_per_feature)

    def set_num_prompts(self, num_prompts):
        """Updates the number of allowed prompts."""
        self.num_prompt_template = num_prompts

    @staticmethod
    def _load_prompt_cache(prompt_cache_location: str):
        """Loads the prompt cache if it exists, otherwise creates a new one."""
        if os.path.isfile(prompt_cache_location):
            with open(prompt_cache_location, 'rb') as f:
                cache = pkl.load(f)
        else:
            cache = {}
        return cache

    def save_prompt_cache(self):
        """Saves a prompt cache, overwiting what is there."""
        with open(self.prompt_cache_location, 'wb') as f:
            pkl.dump(self.prompt_cache, f)

    def get_embedding(self, prompts: list[str], save_cache: bool = False):
        """Gets embeddings of prompts.

        This routine implements getting prompt embeddings. It also implements a
        cache for the prompts, so the embedded prompts that have already been
        generated get queried from here, greatly speeding things up.

        Arguments:
            prompts: The prompts to encode.
            save_cache: Save the cache after getting the embedding and adding it
                        to the cache.
        Returns:
            embeddings: The embedded prompts.
        """
        embeddings = np.zeros((len(prompts), self.embedding_dim))
        miss_indices, miss_prompts = [], []

        # Find cache hits, store prompts that must be generated
        for i, p in enumerate(prompts):
            if p in self.prompt_cache:
                emb = self.prompt_cache[p]
                embeddings[i] = emb
            else:
                miss_indices.append(i)
                miss_prompts.append(p)

        # Encode cache misses, if they exist
        if len(miss_prompts) > 0:
            app.logger.info(f"Missed {len(miss_prompts)} prompts in cache...")
            # Store all prompts on cpu
            encoded_misses = self.sentence_emb_model.encode(miss_prompts)

            # Save encoded misses in correct place, add to cache
            for i, p in enumerate(encoded_misses):
                embeddings[miss_indices[i]] = p
                self.prompt_cache[miss_prompts[i]] = p

        # Restore cache to the correct size
        if len(self.prompt_cache) > self.prompt_cache_size:
            cache_keys = list(self.prompt_cache)
            num_to_remove = len(self.prompt_cache) - self.prompt_cache_size
            keys_to_remove = np.random.choice(
                cache_keys, size=num_to_remove, replace=False)
            for k in keys_to_remove:
                del self.prompt_cache[k]

        if save_cache:
            self.save_prompt_cache()

        return embeddings

    def _fill_wildcard(self,
                       prompts: list[str],
                       wildcard_fname: str,
                       wildcard_vname: str,
                       feature_dict: dict,
                       semantic_feature_names: Union[dict, None]):
        """Creates a new prompt set with a wildcard filled."""
        if len(feature_dict) > 0:
            filled_prompts = set()
            self._do_fill(prompts,
                          wildcard_fname,
                          wildcard_vname,
                          feature_dict,
                          semantic_feature_names,
                          filled_prompts)
            return list(filled_prompts)
        else:
            return prompts

    def _do_fill(self,
                 prompts: list[str],
                 wildcard_fname: str,
                 wildcard_vname: str,
                 feature_dict: dict,
                 semantic_feature_names: Union[dict, None],
                 filled_prompts: set,
                 down_sample: bool = False,
                 num_down_sample: int = 2,
                 split_feature_names: bool = True):
        """Enumerates a wildcard across the prompts."""
        feature_dict_keys = sorted(list(feature_dict.keys()))
        # recursions can grow in size greatly if we don't down sample
        # this enables down-samples the features that are substituted in
        # making the substitution more efficient
        if down_sample:
            # make sure down samples don't exceed num features
            size = min(num_down_sample, len(feature_dict_keys))
            feature_dict_keys = np.random.choice(feature_dict_keys,
                                                 size=size,
                                                 replace=False)

        # Feature names can be compressed into long strings, which may negatively
        # affect parsing ability. We can use wordninja to break them into more
        # semantically meaningful sentences
        if split_feature_names and semantic_feature_names is not None:
            formatted_feature_names = semantic_feature_names
        else:
            formatted_feature_names = None

        for feature in feature_dict_keys:
            for feature_value in feature_dict[feature]:
                feature_value = str(feature_value)
                for prompt in prompts:
                    user_parsed_split = prompt.split('\n')
                    to_rejoin = []

                    if split_feature_names and semantic_feature_names is not None:
                        semantic_feature = formatted_feature_names[feature]
                    else:
                        semantic_feature = feature

                    for i, part in enumerate(user_parsed_split):
                        # user input is given as first part, we just want the semantic
                        # part here
                        if i == 0:
                            part_f = part.replace(wildcard_fname, semantic_feature, 1)
                        else:
                            part_f = part.replace(wildcard_fname, feature, 1)
                        part_f = part_f.replace(
                            wildcard_vname, feature_value, 1)
                        to_rejoin.append(part_f)

                    formatted_prompt = '\n'.join(to_rejoin)

                    if (wildcard_vname not in formatted_prompt and
                            wildcard_fname not in formatted_prompt):
                        filled_prompts.add(formatted_prompt)
                    else:
                        # Recursively fill wildcard if the wildcard
                        # is used multiple times.
                        new_feature_dict = copy.deepcopy(feature_dict)
                        del new_feature_dict[feature]
                        self._do_fill(prompts=[formatted_prompt],
                                      wildcard_fname=wildcard_fname,
                                      wildcard_vname=wildcard_vname,
                                      feature_dict=new_feature_dict,
                                      filled_prompts=filled_prompts,
                                      down_sample=True,
                                      semantic_feature_names=semantic_feature_names)

    @staticmethod
    def _is_valid_prompt(prompt: str):
        """Attempts to catch invalid prompts through several conditions."""
        split_p = prompt.split('\n')
        if not split_p[1].endswith('[E]'):
            return False
        elif not split_p[0].startswith('User: '):
            return False
        elif not split_p[1].startswith('Parsed: '):
            return False
        return True

    @staticmethod
    def _down_sample_features(feature_value_dict: dict,
                              max_values_per_feature: int):
        """Downsamples a values of features."""
        for feature_name in feature_value_dict:
            feature_vals = feature_value_dict[feature_name]
            if len(feature_vals) > max_values_per_feature:
                feature_value_dict[feature_name] = np.random.choice(
                    feature_value_dict[feature_name],
                    replace=False,
                    size=max_values_per_feature)
        return feature_value_dict

    def load_dynamic_prompts(self):
        """Loads the dynamic prompts from file."""

        # Store prompts before filling
        prompts = []

        # Load dynamic prompts
        dynamic_prompt_file_names = find_csv_filenames(
            os.path.join(self.prompt_folder, 'dynamic'), suffix='txt')

        # Load the prompts that are dynamically generated
        filename_to_prompt_ids = {}
        c_prompt_id = 0
        for f in dynamic_prompt_file_names:
            dynamic_fn = os.path.join(self.prompt_folder, 'dynamic', f)
            with open(dynamic_fn, 'r') as file:
                temp_prompts = file.read()
                pre_new_prompt = temp_prompts.split('\n\n')

                new_prompt = self.filter_prompts(pre_new_prompt)

                for prompt in new_prompt:
                    if len(prompt) == 0:
                        warnings.warn(f"Empty prompt from file {f}")

                filename_to_prompt_ids[dynamic_fn] = list(range(c_prompt_id, c_prompt_id + len(new_prompt)))
                c_prompt_id += len(filename_to_prompt_ids[dynamic_fn])
                prompts.extend(new_prompt)

        # Validate prompts
        for prompt in prompts:
            assert self._is_valid_prompt(prompt), 'Invalid prompt %s' % prompt

        # Make sure everything is lowercase
        for i in range(len(prompts)):
            prompts[i] = prompts[i].lower()

        filtered_prompts = prompts

        # Covert to dictionary to more clearly establish ids
        # associated with the prompts
        final_prompts = {i: filtered_prompts[i] for i in range(len(filtered_prompts))}

        return final_prompts, filename_to_prompt_ids

    def filter_prompts(self, pre_new_prompt):
        # Filter prompts if there are no cat or num features, so as not
        # to unnecessarily include these prompts
        do_not_include = []
        if len(self.cat_features) == 0:
            for i, prompt in enumerate(pre_new_prompt):
                if "{cat_features}" in prompt or "{cat_values}" in prompt:
                    do_not_include.append(i)
        if len(self.num_features) == 0:
            for i, prompt in enumerate(pre_new_prompt):
                if "{num_features}" in prompt or "{num_values}" in prompt:
                    do_not_include.append(i)
        new_prompt = []
        for i, prompt in enumerate(pre_new_prompt):
            if i not in do_not_include:
                new_prompt.append(prompt)
        return new_prompt

    def build_filter_filler_dict(self,
                                 cat_features: list,
                                 num_features: list,
                                 cat_filler_dict: dict,
                                 num_filler_dict: dict,
                                 class_dict: dict,
                                 semantic_cat_names: dict,
                                 semantic_num_names: dict,
                                 semantic_class_names: dict) -> dict:
        """Builds a filler dict of filter fillers

        These fillers represent the filtering operations and corresponding parses.

        Args:
            semantic_cat_names:
            semantic_num_names:
            semantic_class_names:
            num_filler_dict:
            class_dict:
            cat_filler_dict:
            num_features:
            cat_features:
        """
        with open(self.filter_filler_dict_loc, "r") as file:
            filter_fillers = file.read()

        all_filter_fillers = filter_fillers.split("\n\n")

        filter_filler_d = {}
        for cur_filter_filler in all_filter_fillers:

            # wild card filling requires list inputs
            cur_filter_filler = [cur_filter_filler]

            if len(cat_features) > 0:
                # Fill the categorical feature wildcards
                cur_filter_filler = self._fill_wildcard(cur_filter_filler,
                                                        '{cat_features}',
                                                        '{cat_values}',
                                                        cat_filler_dict,
                                                        semantic_cat_names)

            if len(num_features) > 0:
                # Fill the numerical feature wildcards
                cur_filter_filler = self._fill_wildcard(cur_filter_filler,
                                                        '{num_features}',
                                                        '{num_values}',
                                                        num_filler_dict,
                                                        semantic_num_names)

            cur_filter_filler = self._fill_wildcard(cur_filter_filler,
                                                    '{non_semantic_class_names}',
                                                    '{class_names}',
                                                    class_dict,
                                                    semantic_class_names)

            for this_filler in cur_filter_filler:
                split_filler = this_filler.split("\n")
                if ('{cat_features}' in split_filler[0] or
                        '{num_features}' in split_filler[0]):
                    continue
                filter_filler_d[split_filler[0]] = [split_filler[1]]

        return filter_filler_d

    def generate_prompts(self,
                         cat_features: list[str],
                         num_features: list[str],
                         target: list[str],
                         class_names: dict,
                         feature_value_dict: dict,
                         max_values_per_feature: int = 10,
                         max_prompts_per_template: int = 300,
                         seed: int = 0):
        """Generates the candidate prompts.

        This routine implements generating the candidate prompt set. It uses
        the categorical and numerical feature values and their associated vals
        to generate many potential prompts.

        Prompts are written as wildcards. Meaning, prompts have values that can
        be substituted in with, for example, categorical feature names. This
        routine facilitates performing this enumeration. The routine enumerates
        all the potential prompts, considering the wildcards and their possible
        values.

        This method saves the generated prompt set in the self.final_prompt_set
        attribute for later use.

        Arguments:
            down_sample_pct:
            seed: Random seed
            max_prompts_per_template: The maximum number of prompts to allow per template.
                                      For templates with more prompts than this number, we
                                      down-sample randomly. Setting this value too high can
                                      make templates with many prompts too heavily weighted
                                      in the data, reducing the effectiveness of the training
                                      and significantly slowing embedding down.
            class_names: A dictionary mapping class indices to semantic names.
            target: [Redundant!] an array containing the class indices (e.g., [0, 1]). This is
                    redundant with class_names though and should be refactored.
            cat_features: The categorical feature names.
            num_features: The numerical feature names.
            feature_value_dict: A dictionary mapping feature names to values.
            max_values_per_feature: The max number of values to generate
                                    prompts from for any given feature.
        """
        np.random.seed(seed)

        # Down-sample features, if max_values_per_feature is set
        if max_values_per_feature is not None:
            feature_value_dict = self._down_sample_features(feature_value_dict,
                                                            max_values_per_feature)

        app.logger.info("Loading dynamic prompts...")

        # Load the dynamic prompts from file
        prompt_set, filename_to_prompt_ids = self.load_dynamic_prompts()

        # Set filename to prompt id as class method
        self.filename_to_prompt_id = filename_to_prompt_ids

        # Build dictionaries that contain the wildcard name and value
        cat_filler_dict = {
            cn: feature_value_dict[cn.lower()] for cn in cat_features}
        num_filler_dict = {
            nn: feature_value_dict[nn.lower()] for nn in num_features}
        exp_dict = {
            'feature importance': ['lime', 'shap', 'feature importance']
        }

        app.logger.info("Building filter dicts...")

        non_semantic_classes = list(class_names.keys())
        # wildcard dictionary to add if classes are needed
        class_dict = {
            'class': [class_names[f] for f in non_semantic_classes]
        }

        full_class_dict = {str(ns_name): [class_names[ns_name]] for ns_name in class_names}

        # add id to categorical values
        cat_filler_dict['id'] = list(np.random.choice(
            len(target), size=max_values_per_feature))

        semantic_cat_names = build_semantic_dict(cat_filler_dict)
        semantic_num_names = build_semantic_dict(num_filler_dict)
        semantic_class_names = build_semantic_dict(full_class_dict)

        # This filler dict contains filtering text and parse info
        filter_filler_dict = self.build_filter_filler_dict(cat_features,
                                                           num_features,
                                                           cat_filler_dict,
                                                           num_filler_dict,
                                                           class_dict=full_class_dict,
                                                           semantic_cat_names=semantic_cat_names,
                                                           semantic_num_names=semantic_num_names,
                                                           semantic_class_names=semantic_class_names)

        # Will contain prompt ids -> prompts with wildcards enumerated
        filled_prompt_set = {}

        app.logger.info("Filling prompt set...")

        # Enumerate the wildcards for each prompt
        for prompt_id in tqdm(prompt_set):
            cur_prompt = [prompt_set[prompt_id]]

            if len(cat_features) > 0:
                # Fill the categorical feature wildcards
                cur_prompt = self._fill_wildcard(cur_prompt,
                                                 '{cat_features}',
                                                 '{cat_values}',
                                                 cat_filler_dict,
                                                 semantic_cat_names)

            if len(num_features) > 0:
                # Fill the numerical feature wildcards
                cur_prompt = self._fill_wildcard(cur_prompt,
                                                 '{num_features}',
                                                 '{num_values}',
                                                 num_filler_dict,
                                                 semantic_num_names)

            if len(filter_filler_dict) > 0:
                # Add the filter text
                cur_prompt = self._fill_wildcard(cur_prompt,
                                                 '{filter_text}',
                                                 '{filter_parse}',
                                                 filter_filler_dict,
                                                 semantic_feature_names=None)

            # Fill the explanation type wildcards
            cur_prompt = self._fill_wildcard(cur_prompt,
                                             '{exp_type}',
                                             '{exp_name}',
                                             exp_dict,
                                             semantic_feature_names=None)

            cur_prompt = self._fill_wildcard(cur_prompt,
                                             '{unused}',
                                             '{class_names}',
                                             class_dict,
                                             semantic_feature_names=None)

            filled_prompt_set[prompt_id] = cur_prompt

        final_prompt_set = {}
        filled_prompt_keys = list(filled_prompt_set.keys())
        filled_prompt_keys = sorted(filled_prompt_keys)

        for prompt_id in tqdm(filled_prompt_keys):
            filled_prompts = filled_prompt_set[prompt_id]

            if len(filled_prompts) > max_prompts_per_template:

                # NOTE: There's something up with the filled prompt sets. They appear in
                # different orderings. I can't figure out why this is the case, so I'm sorting
                # them here, so they always appear in the same ordering for the sake of the
                # random prompt selection. This shouldn't be necessary though.
                sorted_filled_prompts = sorted(filled_prompts)
                filled_prompts = np.random.choice(sorted_filled_prompts,
                                                  replace=False,
                                                  size=max_prompts_per_template)

            if len(filled_prompts) == 0:
                warnings.warn(f"prompt id {prompt_id} has nothing in it! skipping over it")
                continue

            # Extract just user input parts of utterances
            user_utterances = []
            for p in filled_prompts:
                user_part = p.split('\n')[0].split('user: ')[1]
                user_utterances.append(user_part)

            embeddings = torch.tensor(
                self.get_embedding(user_utterances)).float()

            final_prompt_set[prompt_id] = {
                'prompts': filled_prompts,
                'embeddings': embeddings
            }

        self.save_prompt_cache()
        self.final_prompt_set = final_prompt_set

    def get_k_nearest_prompts(self,
                              query: str,
                              metric: str = 'cosine',
                              ordering: str = 'ascending',
                              get_nearest_neighbor: bool = False):
        """Gets the k the nearest prompts.

        This routine implements getting the nearest prompts for a given user
        query. The strategy is to select the prompt templates that are most
        relevant to the query---the number of prompt templates is specified
        by num_prompts---and then select the most relevant prompts from that
        template---specified by num_per_prompt.

        Arguments:
            get_nearest_neighbor:
            ordering:
            query: The query sentence we're finding the prompts for.
            metric: The metric to compute similarity with.
        Return:
            k_nearest_prompts: The k nearest prompts."""

        # Case where no prompts are desired
        if self.num_prompt_template == 0 or self.num_per_knn_prompt_template == 0:
            return []

        if metric == 'random':
            choices = np.random.choice(list(self.final_prompt_set.keys()),
                                       size=(self.num_per_knn_prompt_template *
                                             self.num_prompt_template))
            random_prompt_set = [self.final_prompt_set[c]['prompts'][0] for c in choices]
            return random_prompt_set

        # Doing NN on CPU
        app.logger.info("getting embeddings")
        app.logger.info(query.lower())
        encoded_query = self.sentence_emb_model.encode(
            query.lower(), convert_to_tensor=True).reshape(1, -1).cpu()

        app.logger.info("did embeddings")
        k_nearest_per_prompt = []
        all_distances = []
        for prompt_id in self.final_prompt_set:
            if metric == 'cosine' or metric == 'dot':
                embeddings = self.final_prompt_set[prompt_id]['embeddings']
                distances = self.decide_metric(embeddings, encoded_query, metric)
                closest = np.argsort(distances)[-self.num_per_knn_prompt_template:]

                all_distances.append(distances[closest])
                kn = []
                for c in closest:
                    kn.append(
                        self.final_prompt_set[prompt_id]['prompts'][c])
                k_nearest_per_prompt.append(kn)
            else:
                raise NotImplementedError
        app.logger.info("got closest")
        mean_distances_per_prompt_set = np.array(
            [np.mean(d) for d in all_distances])
        closest = np.argsort(mean_distances_per_prompt_set)[-self.num_prompt_template:]

        out = []
        for c in closest:
            out.extend(k_nearest_per_prompt[c])

        if get_nearest_neighbor:
            return out[-1]

        if ordering == 'ascending':
            return out
        elif ordering == 'descending':
            return out[::-1]
        elif ordering == 'shuffle':
            shuffle(out)
            return out
        else:
            raise NameError(f"Unknown ordering {ordering}")

    @staticmethod
    def decide_metric(embeddings, encoded_query, metric):
        """Computes the similarity depending on the different metrics"""
        if metric == 'dot':
            distances = util.dot_score(
                encoded_query, embeddings).cpu().numpy()[0]
        else:
            distances = util.pytorch_cos_sim(
                encoded_query, embeddings).cpu().numpy()[0]
        return distances

    @staticmethod
    def _strip_numerical_values(query: str):
        options = []

        # remove punctuation that could cause numbers to be parsed
        # incorrectly
        query = query.replace("!", "")
        query = query.replace("?", "")
        query = query.replace(",", "")
        query = query.replace("+", "")
        query = query.replace("#", "")
        query = query.replace("(", "")
        query = query.replace(")", "")

        # Don't remove decimals in the middle of sentences...
        if query.endswith("."):
            query = query[:-1]

        words_to_check_as_nums = []
        words_to_check_as_nums.extend(query.split(' '))

        # Get cases that have letters attached to numbers
        for item in query.split(' '):
            for char in string.ascii_lowercase:
                remove_text = item.replace(char, "")
                if remove_text != "":
                    words_to_check_as_nums.append(remove_text)

        for word in words_to_check_as_nums:
            # Note(dylan): Is there a cleaner way to do these checks?
            # Feels somewhat ugly to me, but ¯\_(ツ)_/¯
            try:
                # Check if can be converted to float
                float(word)
                options.append(word)
                continue
            except ValueError:
                pass

            try:
                # Check if it's a text word
                number = w2n.word_to_num(word)
                options.append(str(number))
            except ValueError:  # noqa: E722
                pass
        return options

    def _extract_id_nums(self, query: str):
        """Extracts any numbers in the query string that could be ids.

        We augment the grammar with any potential data id values that appear
        in the question. We do this because there are often many items in the
        data and including all the query id's in the grammar is not so
        effective, causing large slowdowns. So, we just include potential ids found
        in the query on-the-fly.

        Arguments:
            query: the user query
        Returns:
            nonterminal: a new nonterminal containing any id values. If there aren't
                         any, returns None.
        """
        options = self._strip_numerical_values(query)

        if len(options) == 0:
            return {}

        string = ""
        for op in options:
            string += "\" id " + op + "\"" + " |"
        string = string[:-1]

        return {"id": string}

    def _extract_numerical_values(self, query: str):
        """Extracts any numerical values in the string.

        This finds any exact numerical values in this string (i.e., 123 and *not* one hundred and
        twenty three) and adds them to a nonterminal called adhocnumvalues. This nonterminal is used
        later on for handling numerical inputs to the system.

        NOTE(dylan): This could be updated for more advanced processing of the numerical values. For instance,
        this routine could be made to handle cases like "one hundred and thirty" --> 130. Talking to matt
        gardner a bit he said this should be easy to do, but I haven't yet found the right package.

        Arguments:
            query: the user query to mine the numbers from
        Returns:
            nonterminal: a new nonterminal containing any numerical values. If there aren't
                         any, returns None.
        """
        options = self._strip_numerical_values(query)

        temp_string = "\" unknown\" |"
        for op in options:
            temp_string += "\" " + op + "\"" + " |"
        temp_string = temp_string[:-1]

        return {"adhocnumvalues": temp_string}

    def get_prompts(self,
                    query: str,
                    metric: str = 'cosine',
                    ordering: str = 'ascending',
                    error_analysis: bool = False):
        """Gets the prompts given the query."""

        if self.skip:
            selected_prompts = ""
        else:
            selected_prompts = self.get_k_nearest_prompts(
                query, metric=metric, ordering=ordering)

        app.logger.info(f'Selected prompts {selected_prompts}')

        # Format query
        if len(selected_prompts) > 0:
            joined_prompts = '\n\n'.join(selected_prompts)
            joined_prompts += '\n\n'
        else:
            joined_prompts = ""

        joined_prompts += f'User: {query}\nParsed:'
        joined_prompts = joined_prompts.lower()
        joined_prompts = joined_prompts.replace("user:", "input:")

        id_adhoc = self._extract_id_nums(query)
        num_adhoc = self._extract_numerical_values(query)

        if error_analysis:
            return joined_prompts, {**id_adhoc, **num_adhoc}, selected_prompts
        return joined_prompts, {**id_adhoc, **num_adhoc}
