"""This file implements objects that generate and cache explanations.

The main caveat so far is that sklearn models are currently the only types
of models supported.
"""
import os
import pickle as pkl
import warnings
from typing import Union, Any

import pandas as pd
from tqdm import tqdm
from typing import Callable

from flask import Flask
import gin
import numpy as np

import dice_ml

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
class TabularDice(Explanation):
    """Tabular dice counterfactual explanations."""

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 num_features: list[str],
                 num_cfes_per_instance: int = 10,
                 num_in_short_summary: int = 3,
                 desired_class: str = "opposite",
                 cache_location: str = "./cache/dice-tabular.pkl",
                 class_names: dict = None):
        """Init.

        Arguments:
            model: the sklearn style model, where model.predict(data) returns the predictions
                   and model.predict_proba returns the prediction probabilities
            data: the pandas df data
            num_features: The *names* of the numerical features in the dataframe
            num_cfes_per_instance: The total number of cfes to generate per instance
            num_in_short_summary: The number of cfes to include in the short summary
            desired_class: Set to "opposite" to compute opposite class
            cache_location: Location to store cache.
            class_names: The map between class names and text class description.
        """
        super().__init__(cache_location, class_names)
        self.temp_outcome_name = 'y'
        self.model = self.wrap(model)
        self.num_features = num_features
        self.desired_class = desired_class
        self.num_cfes_per_instance = num_cfes_per_instance
        self.num_in_short_summary = num_in_short_summary

        self.dice_model = dice_ml.Model(model=self.model, backend="sklearn")

        # Format data in dice accepted format
        predictions = self.model.predict(data)
        if self.model.predict_proba(data).shape[1] > 2:
            self.non_binary = True
        else:
            self.non_binary = False
        data[self.temp_outcome_name] = predictions

        self.classes = np.unique(predictions)
        self.dice_data = dice_ml.Data(dataframe=data,
                                      continuous_features=self.num_features,
                                      outcome_name=self.temp_outcome_name)

        data.pop(self.temp_outcome_name)

        self.exp = dice_ml.Dice(
            self.dice_data, self.dice_model, method="random")

    def wrap(self, model: Any):
        """Wraps model, converting pd to df to silence dice warnings"""
        class Model:
            def __init__(self, m):
                self.model = m

            def predict(self, X):
                return self.model.predict(X.values)

            def predict_proba(self, X):
                return self.model.predict_proba(X.values)
        return Model(model)

    def run_explanation(self,
                        data: pd.DataFrame,
                        desired_class: str = None):
        """Generate tabular dice explanations.

        Arguments:
            data: The data to generate explanations for in pandas df.
            desired_class: The desired class of the cfes. If None, will use the default provided
                           at initialization.
        Returns:
            explanations: The generated cf explanations.
        """
        if self.temp_outcome_name in data:
            raise NameError(f"Target Variable {self.temp_outcome_name} should not be in data.")

        if desired_class is None:
            desired_class = self.desired_class

        cfes = {}
        for d in tqdm(list(data.index)):
            # dice has a few function calls that are going to be deprecated
            # silence warnings for ease of use now
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.non_binary and desired_class == "opposite":
                    desired_class = int(np.random.choice([p for p in self.classes if p != self.model.predict(data.loc[[d]])[0]]))
                cur_cfe = self.exp.generate_counterfactuals(data.loc[[d]],
                                                            total_CFs=self.num_cfes_per_instance,
                                                            desired_class=desired_class)
            cfes[d] = cur_cfe
        return cfes

    def get_change_string(self, cfe: Any, original_instance: Any):
        """Builds a string describing the changes between the cfe and original instance."""
        cfe_features = list(cfe.columns)
        original_features = list(original_instance.columns)
        message = "CFE features and Original Instance features are different!"
        assert set(cfe_features) == set(original_features), message

        change_string = ""
        for feature in cfe_features:
            orig_f = original_instance[feature].values[0]
            cfe_f = cfe[feature].values[0]

            if isinstance(cfe_f, str):
                cfe_f = float(cfe_f)

            if orig_f != cfe_f:
                if cfe_f > orig_f:
                    inc_dec = "increase"
                else:
                    inc_dec = "decrease"
                change_string += f"{inc_dec} {feature} to {str(round(cfe_f, self.rounding_precision))}"
                change_string += " and "
        # Strip off last and
        change_string = change_string[:-5]
        return change_string

    def summarize_explanations(self,
                               data: pd.DataFrame,
                               ids_to_regenerate: list[int] = None,
                               filtering_text: str = None,
                               save_to_cache: bool = False):
        """Summarizes explanations for dice tabular.

        Arguments:
            data: pandas df containing data.
            ids_to_regenerate:
            filtering_text:
            save_to_cache:
        Returns:
            summary: a string containing the summary.
        """

        if ids_to_regenerate is None:
            ids_to_regenerate = []
        if data.shape[0] > 1:
            return ("", "I can only compute how to flip predictions for single instances at a time."
                    " Please narrow down your selection to a single instance. For example, you"
                    " could specify the id of the instance to want to figure out how to change.")

        ids = list(data.index)
        key = ids[0]

        explanation = self.get_explanations(ids,
                                            data,
                                            ids_to_regenerate=ids_to_regenerate,
                                            save_to_cache=save_to_cache)
        original_prediction = self.model.predict(data)[0]
        original_label = self.get_label_text(original_prediction)

        cfe = explanation[key]
        final_cfes = cfe.cf_examples_list[0].final_cfs_df
        final_cfe_ids = list(final_cfes.index)

        if self.temp_outcome_name in final_cfes.columns:
            final_cfes.pop(self.temp_outcome_name)

        new_predictions = self.model.predict(final_cfes)

        original_instance = data.loc[[key]]

        if filtering_text is not None and len(filtering_text) > 0:
            filtering_description = f"For instances where <b>{filtering_text}</b>"
        else:
            filtering_description = ""
        output_string = f"{filtering_description}, the original prediction is "
        output_string += f"<em>{original_label}</em>. "
        output_string += "Here are some options to change the prediction of this instance."
        output_string += "<br><br>"

        additional_options = "Here are some more options to change the prediction of"
        additional_options += f" instance id {str(key)}.<br><br>"

        output_string += "First, if you <em>"
        transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]

        for i, c_id in enumerate(final_cfe_ids):
            # Stop the summary in case its getting too large
            if i < self.num_in_short_summary:
                if i != 0:
                    output_string += f"{np.random.choice(transition_words)} if you <em>"
                output_string += self.get_change_string(final_cfes.loc[[c_id]], original_instance)
                new_prediction = self.get_label_text(new_predictions[i])
                output_string += f"</em>, the model will predict {new_prediction}.<br><br>"
            else:
                additional_options += "If you <em>"
                additional_options += self.get_change_string(final_cfes.loc[[c_id]], original_instance)
                new_prediction = self.get_label_text(new_predictions[i])
                additional_options += f"</em>, the model will predict {new_prediction}.<br><br>"

        output_string += "If you want some more options, just ask &#129502"

        return additional_options, output_string


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
                 use_selection: bool = True):
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
        np_data = data.to_numpy()
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
                                      include_confidence_text: bool = False):
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
        if ids_to_regenerate is None:
            ids_to_regenerate = []
        ids = list(data.index)

        # Note that the explanations are returned as MegaExplanation
        # dataclass instances
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

        full_summary, short_summary = self.format_explanations_to_string(feature_importances,
                                                                         scores,
                                                                         filtering_text)
        return full_summary, short_summary
