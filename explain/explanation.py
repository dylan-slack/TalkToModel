"""This file implements objects that generate and cache explanations.

The main caveat so far is that sklearn models are currently the only types
of models supported.
"""
import os
import pickle as pkl
from typing import Union

import pandas
import pandas as pd
from tqdm import tqdm
from typing import Callable

from flask import Flask
import gin
import numpy as np

from explain.mega_explainer.explainer import Explainer

app = Flask(__name__)


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        cache = {}
    return cache


@gin.configurable
class Explanation:
    """A top level class defining explanations."""

    def __init__(self,
                 cache_location: str,
                 class_names: dict = None,
                 max_cache_size: int = 1_000_000,
                 rounding_precision: int = 3):
        """Init.

        Arguments:
            cache_location:
            class_names:
            max_cache_size:
        """
        self.max_cache_size = max_cache_size
        self.cache_loc = cache_location
        self.class_names = class_names
        self.cache = load_cache(cache_location)
        self.rounding_precision = rounding_precision

    def get_label_text(self, label: int):
        """Gets the label text."""
        if self.class_names is not None:
            label_name = self.class_names[label]
        else:
            label_name = str(label)
        return label_name

    def update_cache_size(self, new_cache_size: int):
        """Change the size of the cache."""
        self.max_cache_size = new_cache_size

    def _cache_size(self):
        return len(self.cache)

    def _save_cache(self):
        """Saves the current self.cache."""
        with open(self.cache_loc, 'wb') as file:
            pkl.dump(self.cache, file)

    def _get_from_cache(self, ids: list[int], ids_to_regenerate: list[int] = None):
        if ids_to_regenerate is None:
            ids_to_regenerate = []
        misses, hits = [], {}
        for c_id in ids:
            if (c_id not in self.cache) or (c_id in ids_to_regenerate):
                misses.append(c_id)
            else:
                hits[c_id] = self.cache[c_id]
        app.logger.info(f"Missed {len(misses)} items in cache lookup")
        return misses, hits

    def _write_to_cache(self, expls: dict):
        """Writes explanations to cache at ids, overwriting."""
        for i, c_id in enumerate(expls):
            self.cache[c_id] = expls[c_id]

        # resize if we exceed cache, I haven't tested this currently
        while len(self.cache) > self.max_cache_size:
            keys = list(self.cache)
            to_remove = np.random.choice(keys)
            del self.cache[to_remove]

        self._save_cache()

    def get_explanations(self,
                         ids: list[int],
                         data: pd.DataFrame,
                         ids_to_regenerate: list[int] = None,
                         save_to_cache: bool = True):
        """Gets explanations corresponding to ids in data, where data is a pandas df.

        This routine will pull explanations from the cache if they exist. If
        they don't it will call run_explanation on these ids.
        """
        if ids_to_regenerate is None:
            ids_to_regenerate = []
        ids_to_gen, hit_expls = self._get_from_cache(ids, ids_to_regenerate)
        if len(ids_to_gen) > 0:
            rest_of_exp = self.run_explanation(data.loc[ids_to_gen])
            if save_to_cache:
                self._write_to_cache(rest_of_exp)

            hit_expls = {**hit_expls, **rest_of_exp}

        return hit_expls

    def __repr__(self):
        output = "Loaded explanation.\n"
        output += f"  *cache of size {self._cache_size()}\n"
        output += f"  *cache located in {self.cache_loc}"
        return output


@gin.configurable
class MegaExplainer(Explanation):
    """Generates many model agnostic explanations and selects the best one.

    Note that this class can be used to recover a single explanation as well
    by setting the available explanations to the particular one, i.e., 'lime'
    """

    def __init__(self,
                 prediction_fn: Callable[[np.ndarray], np.ndarray],
                 data: pd.DataFrame,
                 cat_features: Union[list[int], list[str]],
                 cache_location: str = "./cache/mega-explainer-tabular.pkl",
                 class_names: list[str] = None,
                 use_selection: bool = True,
                 categorical_mapping: dict = None):
        """Init.

        Args:
            prediction_fn: A callable function that computes the prediction probabilities on some
                           data.
            data:
            cat_features:
            cache_location:
            class_names:
        """
        super().__init__(cache_location, class_names)
        self.prediction_fn = prediction_fn
        self.data = data

        cat_features = self.get_cat_features(data, cat_features)

        # Initialize the explanation selection
        self.mega_explainer = Explainer(explanation_dataset=self.data.to_numpy(),
                                        explanation_model=self.prediction_fn,
                                        feature_names=data.columns,
                                        discrete_features=cat_features,
                                        use_selection=use_selection)
        self.categorical_mapping = categorical_mapping

    @staticmethod
    def get_cat_features(data: pd.DataFrame,
                         cat_features: Union[list[int], list[str]]) -> list[int]:
        """Makes sure categorical features are list of indices.

        If not, will convert list of feature names to list of indices.

        Args:
            data: The dataset given as pd.DataFrame
            cat_features: Either a list of indices or feature names. If given as a list of feature
                          names, it will be converted to list of indices.
        Returns:
            cat_features: The list of categorical feature indices.
        """
        if all([isinstance(c, str) for c in cat_features]):
            feature_names = list(data.columns)
            new_cat_features = []
            for c in cat_features:
                new_cat_features.append(feature_names.index(c))
            cat_features = new_cat_features
        else:
            message = "Must be list of indices for cat features or all str\n"
            message += "feature names (which we will convert to indices)."
            assert all([isinstance(c, int) for c in cat_features]), message
        return cat_features

    def run_explanation(self, data: pd.DataFrame):
        """Generate mega explainer explanations

        Arguments:
            data: The data to compute explanations on of shape (n_instances, n_features).
        Returns:
            generated_explanations: A dictionary containing {id: explanation} pairs
        """
        generated_explanations = {}
        np_data = data.to_numpy(dtype=object)
        # Generate the lime explanations
        pbar = tqdm(range(np_data.shape[0]))
        for i in pbar:
            pbar.set_description(f"Processing explanation selection {i}")
            # Right now just explaining the top label
            output = self.mega_explainer.explain_instance(np_data[i])
            # Make sure we store id of data point from reference pandas df
            generated_explanations[list(data.index)[i]] = output
        return generated_explanations

    @staticmethod
    def format_option_text(sig: list[float], i: int):
        """Formats a cfe option."""
        shortened_output = ""

        if sig[1] > 0:
            pos_neg = "positive"
        else:
            pos_neg = "negative"

        if i == 0:
            shortened_output += (f"<b>{sig[0]}</b> is the <b>most important </b>feature and has a"
                                 f" <em>{pos_neg}</em> influence on the predictions")

        if i == 1:
            shortened_output += (f"<b>{sig[0]}</b> is the <b>second</b> most important feature and has a"
                                 f" <em>{pos_neg}</em> influence on the predictions")
        if i == 2:
            shortened_output += (f"<b>{sig[0]}</b> is the <b>third</b> most important feature and has a"
                                 f" <em>{pos_neg}</em> influence on the predictions")

        return shortened_output, pos_neg

    def format_explanations_to_string(self,
                                      feature_importances: dict,
                                      scores: dict, filtering_text: str,
                                      include_confidence_text: bool = False,
                                      feature_values: pandas.DataFrame = None):
        """Formats dict of label -> feature name -> feature_importance dicts to string.

        TODO(dylan): In shortened text summary, consider adding something about anomalous
                     feature importances, i.e., sometimes race is important.

        Arguments:
            include_confidence_text: Add text to describe the accuracy of the explanation(s).
            feature_importances: A dictionary that contains the mapping label -> feature name
                                 -> feature importance.
            scores: A dictionary with mapping label -> fidelity score. This is used to summarize
                    predictive performance across a
            filtering_text: text describing the filtering operations for the data the explanations
                            are run on.
            feature_values: Dataframe of the instance that is explained.
        Returns:
            r_str: Two string, the first providing a full summary of the result and the second providing
                   a condensed summary of the result.
        """
        full_print_out = ""
        shortened_output = ""
        for label in feature_importances:
            sig_coefs = []
            if self.class_names is not None:
                label_name = self.class_names[label]
            else:
                label_name = str(label)

            if filtering_text is not None and len(filtering_text) > 0:
                starter_text = f"For instances with <b>{filtering_text}</b> predicted <em>{label_name}</em>:"
            else:
                starter_text = f"For all the instances predicted <em>{label_name}</em>"

            full_print_out += starter_text
            shortened_output += starter_text

            full_print_out += " the feature importances are:<br>"

            for feature_imp in feature_importances[label]:
                if feature_values is not None and self.categorical_mapping is not None:
                    feature_value = feature_values[feature_imp].values[0]
                    column_id = self.data.columns.get_loc(feature_imp)
                    try:
                        decoded_feature_value = self.categorical_mapping[str(column_id)][int(feature_value)]
                    except KeyError:
                        decoded_feature_value = feature_value
                    sig_coefs.append([feature_imp + " = " + str(decoded_feature_value),
                                      np.mean(feature_importances[label][feature_imp])])
                else:
                    sig_coefs.append([feature_imp,
                                      np.mean(feature_importances[label][feature_imp])])

            sig_coefs.sort(reverse=True, key=lambda x: abs(x[1]))
            app.logger.info(sig_coefs)

            # Add full list of feature importances to the comprehensive print out
            shortened_output += "<ul>"
            for i, sig in enumerate(sig_coefs):
                new_text, pos_neg = self.format_option_text(sig, i)
                if new_text != "":
                    shortened_output += "<li>" + new_text + "</li>"
                feature_imp = str(round(sig[1], self.rounding_precision))
                full_print_out += f"<br>{sig[0]} ({pos_neg} influence {feature_imp})"
            shortened_output += "</ul>"

            full_print_out += "<br><br>"

            # Add the accuracy rating
            score = np.median(scores[label])
            if score > 0.8:
                conf = 'very accurate'
            elif score > 0.4:
                conf = 'moderately accurate'
            else:
                conf = 'not that accurate'

            format_r2 = str(round(score, self.rounding_precision))

            if include_confidence_text:
                confidence_text = " These explanations fit the model with an average"
                confidence_text += f" R2 score of {format_r2}, meaning they are {conf}."
            else:
                confidence_text = ""

            full_print_out += confidence_text
            shortened_output += confidence_text

            full_print_out += "<br><br>"
            shortened_output += "<br><br>"

        shortened_output += "I can provide a more comprehensive overview of how important"
        shortened_output += " different features in the data are for the model's predictions, just"
        shortened_output += " ask for more description &#129502<br><br>"

        return full_print_out, shortened_output

    def get_feature_importances(self,
                                data: pd.DataFrame,
                                ids_to_regenerate: list = None,
                                save_to_cache=False):
        """
        Arguments:
            data: pandas df containing data.
            ids_to_regenerate: ids of instances to regenerate explanations for even if they're cached
            filtering_text: text describing the filtering operations for the data the explanations
                            are run on.
            save_to_cache: whether to write explanations generated_to_cache. If ids are regenerated and
                           save_to_cache is set to true, the existing explanations will be overwritten.
        Returns:
            summary: a string containing the summary.
        """

        # Note that the explanations are returned as MegaExplanation
        # dataclass instances
        if ids_to_regenerate is None:
            ids_to_regenerate = []
        ids = list(data.index)

        explanations = self.get_explanations(ids,
                                             data,
                                             ids_to_regenerate=ids_to_regenerate,
                                             save_to_cache=save_to_cache)

        # Keep a dictionary of the different labels being
        # explained and the associated feature importances.
        # This dictionary maps label -> feature name -> feature importances
        # Doing this makes it easy to summarize explanations across different
        # predicted classes.
        feature_importances = {}

        # The same as above except for scores
        scores = {}
        for i, current_id in enumerate(ids):

            # store the coefficients of the explanation
            label = explanations[current_id].label
            list_exp = explanations[current_id].list_exp

            # Add the label if it is not in the dictionary
            if label not in feature_importances:
                feature_importances[label] = {}

            for tup in list_exp:
                if tup[0] not in feature_importances[label]:
                    feature_importances[label][tup[0]] = []
                feature_importances[label][tup[0]].append(tup[1])

            # also retain the scores
            if label not in scores:
                scores[label] = []
            scores[label].append(explanations[ids[i]].score)
        return feature_importances, scores

    def summarize_explanations(self,
                               data: pd.DataFrame,
                               ids_to_regenerate: list = None,
                               filtering_text: str = None,
                               save_to_cache: bool = False):
        """Summarizes explanations for lime tabular.

        Arguments:
            data: pandas df containing data.
            ids_to_regenerate: ids of instances to regenerate explanations for even if they're cached
            filtering_text: text describing the filtering operations for the data the explanations
                            are run on.
            save_to_cache: whether to write explanations generated_to_cache. If ids are regenerated and
                           save_to_cache is set to true, the existing explanations will be overwritten.
        Returns:
            summary: a string containing the summary.
        """

        feature_importances, scores = self.get_feature_importances(data, ids_to_regenerate, save_to_cache)

        full_summary, short_summary = self.format_explanations_to_string(feature_importances,
                                                                         scores,
                                                                         filtering_text,
                                                                         feature_values=data)
        return full_summary, short_summary
