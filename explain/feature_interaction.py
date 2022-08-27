"""Feature interaction index."""
import copy
from typing import Any

import numpy as np
import pandas as pd
import tqdm


class FeatureInteraction:
    """Feature interaction explainer."""

    def __init__(self,
                 data: pd.DataFrame,
                 prediction_fn: Any,
                 cat_features: list[str],
                 class_ind: int = None,
                 verbose: bool = False):
        """Init.

        Args:
            data: data to compute feature interactions
            prediction_fn: the prediction function
            cat_features: categorical features
            class_ind: the class index to compute the feature interaction effects on
            verbose: whether to enable verbosity
        """
        self.data = data

        self.class_ind = class_ind
        self.prediction_fn = prediction_fn
        self.cat_features = cat_features
        self.verbose = verbose

    def feature_interaction(self,
                            i: str,
                            j: str,
                            sub_sample_pct: float = None,
                            number_sub_samples: int = None):
        """Computes the feature interaction between i and j

        Args:
            i: feature name one
            j: feature name two
            sub_sample_pct: pct to sample down feature's values to make it
                            run quicker. If number_sub_samples is not None, this
                            will be ignored
            number_sub_samples: the number of subsamples to use
        """

        # If number sub_samples is set, use this value
        if number_sub_samples is not None:
            num_sub_samples = number_sub_samples
        else:
            # Otherwise, see if percentage is provided
            if sub_sample_pct is None:
                num_sub_samples = int(len(self.data) * 0.10)
            elif sub_sample_pct == 'full':
                num_sub_samples = len(self.data)
            else:
                sub_sample_pct /= 100
                num_sub_samples = int(len(self.data) * sub_sample_pct)

        i_given_j = self.conditional_interaction(i, j, self.data, num_sub_samples)
        j_given_i = self.conditional_interaction(j, i, self.data, num_sub_samples)
        mean_interaction = np.mean([i_given_j, j_given_i])
        return mean_interaction

    def choose_values_to_sample(self, i: str, data: pd.DataFrame, num_sub_samples: int):
        """Samples down a feature, making marginalization easier

        Returns the sampled down feature vector.
        """

        unique_values = np.sort(data[i].unique())

        if len(unique_values) < num_sub_samples:
            return unique_values

        if i in self.cat_features:
            # Randomly subsample categorical features
            indices = np.random.choice(len(unique_values), size=num_sub_samples)
            samples = unique_values[indices]
        else:
            # For sampling numeric features, take sorted feature and space out indices
            # so sample is more representative
            indices = list(range(0, len(unique_values), len(unique_values) // num_sub_samples))
            samples = unique_values[indices]

        return samples

    def conditional_interaction(self, i: str, j: str, data: pd.DataFrame, num_sub_samples: int):
        """Computes the feature interaction of i conditioned on j"""

        # Choose sub sample of feature
        unique_values_of_j = self.choose_values_to_sample(j, data, num_sub_samples)

        results = []
        if self.verbose:
            progress_bar = tqdm.tqdm(unique_values_of_j)
        else:
            progress_bar = unique_values_of_j
        for unique_val in progress_bar:
            fixed_j_dataset = copy.deepcopy(data)
            fixed_j_dataset[j] = unique_val
            flatness_at_j = self.partial_dependence_flatness(i, fixed_j_dataset, num_sub_samples)
            results.append(flatness_at_j)
        return np.std(results)

    def partial_dependence_flatness(self, i: str, data: pd.DataFrame, num_sub_samples: int) -> float:
        """Computes a notion of flatness of the partial dependence

        This metric is from: https://arxiv.org/pdf/1805.04755.pdf
        """

        if i in self.cat_features:
            _, dependence = self.partial_dependence(i, data, num_sub_samples)
            max_dep, min_dep = np.max(dependence, axis=0), np.min(dependence, axis=0)
            flatness = (max_dep - min_dep) / 4
        else:
            _, dependence = self.partial_dependence(i, data, num_sub_samples)
            mean_dependence = np.mean(dependence, axis=0)

            # The sample std
            flatness = np.sum((dependence - mean_dependence) ** 2) / (len(dependence) - 1)

        # If there are many classes and no label, return the max
        # this could be desirable because it's important to say if
        # interactions exist even if only for one class
        if self.class_ind is not None:
            return np.max(flatness)

        return flatness[self.class_ind]

    def partial_dependence(self, i: str, data: pd.DataFrame, num_sub_samples: int) -> Any:
        """Computes all the partial dependence values for feature i

        Args:
            num_sub_samples:
            i: The feature names to compute partial dependence for
            data:
        Returns:
            pdp: A mapping from a unique value in the column to the average prediction with that value
                 substituted in. Plot these to get the partial dependence.
        """

        unique_column_values = self.choose_values_to_sample(i, data, num_sub_samples)

        pdp = {}
        for unique_value in unique_column_values:
            # substitute unique value into the data frame
            updated_dataset = copy.deepcopy(data)
            updated_dataset[i] = unique_value

            # compute predictions on updated data
            predictions = self.prediction_fn(updated_dataset.to_numpy())
            # compute the average prediction
            average_prediction = np.mean(predictions, axis=0)
            pdp[unique_value] = average_prediction

        feature_vals = np.array(list(pdp.keys()))
        dependence = np.array([pdp[val] for val in feature_vals])
        sorted_vals = np.argsort(feature_vals)

        feature_vals = feature_vals[sorted_vals]
        dependence = dependence[sorted_vals]

        return feature_vals, dependence
