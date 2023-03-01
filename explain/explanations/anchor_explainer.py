import warnings

import gin
import numpy as np
import pandas as pd
from anchor import anchor_tabular
from anchor.anchor_explanation import AnchorExplanation
from tqdm import tqdm

from explain.explanation import Explanation


@gin.configurable
class TabularAnchor(Explanation):
    """This class generates ANCHOR explanations for tabular data."""

    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 categorical_names: dict,
                 class_names: dict,
                 feature_names: list,
                 mode: str = "tabular",
                 cache_location: str = "./cache/anchor-tabular.pkl"):
        """

        Args:
            model: The model to explain.
            data: the background dataset provided at pandas df
            categorical_names: map from integer to list of strings, names for each
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal or continuous, and thus discretized.
            class_names: dict of class names
            feature_names: list of feature names
            mode: currently only "tabular" is supported

        """
        super().__init__(cache_location, class_names)
        self.data = data.to_numpy()
        self.mode = mode
        self.model = model
        self.categorical_names = categorical_names
        self.class_names = list(class_names.values())
        self.feature_names = feature_names

        if self.mode == "tabular":
            self.explainer = anchor_tabular.AnchorTabularExplainer(self.class_names,
                                                                   self.feature_names,
                                                                   self.data,
                                                                   self.categorical_names)
        else:
            message = "Currently, only anchor tabular explainer is implemented"
            raise NotImplementedError(message)

    def get_explanation(self, data_x: np.ndarray) -> AnchorExplanation:
        """

        Args:
            data_x: the data instance to explain of shape (1, num_features)
        Returns: Anchor explanation object

        """
        if self.mode == "tabular":
            output = self.explainer.explain_instance(data_x[0],
                                                     self.model.predict,
                                                     threshold=0.85,
                                                     max_anchor_size=3)
            return output

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

        anchors = {}
        for d in tqdm(list(data.index)):
            cur_anchor = self.get_explanation(data.loc[[d]].to_numpy())
            anchors[d] = cur_anchor
        return anchors

    def summarize_explanations(self,
                               data: pd.DataFrame,
                               ids_to_regenerate: list[int] = None,
                               filtering_text: str = None,
                               save_to_cache: bool = False):
        """Summarizes explanations for Anchor tabular.

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
        # Not needed in question selection case
        """if data.shape[0] > 1:
            return ("", "I can only compute Anchors for single instances at a time."
                        " Please narrow down your selection to a single instance. For example, you"
                        " could specify the id of the instance to want to figure out how to change.")"""

        ids = list(data.index)
        key = ids[0]

        explanation = self.get_explanations(ids,
                                            data,
                                            ids_to_regenerate=ids_to_regenerate,
                                            save_to_cache=save_to_cache)
        original_prediction = self.model.predict(data)[0]
        original_label = self.get_label_text(original_prediction)

        exp = explanation[key]

        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())
        print()
        # TODO: Understand coverage and precision for impact of explanation

        original_instance = data.loc[[key]]
        # TODO: Write Better Output explanation
        if filtering_text is not None and len(filtering_text) > 0:
            filtering_description = f"For instances where <b>{filtering_text}</b>"
        else:
            filtering_description = ""
        output_string = f"{filtering_description}, the original prediction is "
        output_string += f"<em>{original_label}</em>. "
        output_string += "By fixing the following attributes, the prediction stays the same even though other attributes are changed:"
        output_string += "<br><br>"

        additional_options = "Here are some more options to change the prediction of"
        additional_options += f" instance id {str(key)}.<br><br>"

        output_string += ' AND '.join(exp.names())
        """transition_words = ["Further,", "Also,", "In addition,", "Furthermore,"]

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

        output_string += "If you want some more options, just ask &#129502"""

        return additional_options, output_string
