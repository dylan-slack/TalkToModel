"""Generates SHAP explanations."""
import numpy as np
import torch
import shap

from explain.mega_explainer.base_explainer import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """The SHAP explainer"""

    def __init__(self,
                 model,
                 data: torch.FloatTensor,
                 link: str = 'identity'):
        """Init.

        Args:
            model: model object
            data: pandas data frame or numpy array
            link: str, 'identity' or 'logit'
        """
        super().__init__(model)

        # Store the data
        self.data = shap.kmeans(data, 25)

        # Use the SHAP kernel explainer in all cases. We can consider supporting
        # domain specific methods in the future.
        self.explainer = shap.KernelExplainer(self.model, self.data, link=link)

    def get_explanation(self, data_x: np.ndarray, label) -> torch.FloatTensor:
        """Gets the SHAP explanation.

        Returns SHAP values as the explanation of the decision made for the input data (data_x)

        Args:
            label: The label to explain.
            data_x: data sample to explain. This sample is of type np.ndarray and is of shape
                    (1, dims).
        Returns:
            final_shap_values: SHAP values [dim (shap_vals) == dim (data_x)]
        """

        # Compute the shapley values on the **single** instance
        shap_vals = self.explainer.shap_values(data_x[0], nsamples=10_000, silent=True)

        # Ensure that we select the correct label, if shap values are
        # computed on output prob. distribution
        if len(shap_vals) > 1:
            shap_value_at_label = shap_vals[label]
            final_shap_values = torch.FloatTensor(shap_value_at_label)
        else:
            final_shap_values = torch.FloatTensor(shap_vals)
        return final_shap_values, 0
