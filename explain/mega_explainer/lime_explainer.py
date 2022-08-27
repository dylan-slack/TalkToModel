"""LIME explanations."""
from lime import lime_tabular
import numpy as np
import torch

from explain.mega_explainer.base_explainer import BaseExplainer


class Lime(BaseExplainer):
    """This class generates LIME explanations for tabular data."""

    def __init__(self,
                 model,
                 data: np.ndarray,
                 discrete_features: list,
                 mode: str = "tabular",
                 sample_around_instance: bool = False,
                 kernel_width: float = 0.75,
                 n_samples: int = 10_000,
                 discretize_continuous: bool = False):
        """

        Args:
            model:
            data: the background dataset provided at np.ndarray
            mode:
            sample_around_instance:
            kernel_width:
            n_samples:
            discretize_continuous:
        """
        self.data = data
        self.mode = mode
        self.model = model
        self.n_samples = n_samples
        self.discretize_continuous = discretize_continuous
        self.sample_around_instance = sample_around_instance
        self.discrete_features = discrete_features

        if self.mode == "tabular":
            self.explainer = lime_tabular.LimeTabularExplainer(self.data,
                                                               mode="classification",
                                                               categorical_features=self.discrete_features,
                                                               sample_around_instance=self.sample_around_instance,
                                                               discretize_continuous=self.discretize_continuous,
                                                               kernel_width=kernel_width * np.sqrt(
                                                                   self.data.shape[1]),
                                                               )
        else:
            message = "Currently, only lime tabular explainer is implemented"
            raise NotImplementedError(message)

        super(Lime, self).__init__(model)

    def get_explanation(self, data_x: np.ndarray, label=None) -> tuple[torch.FloatTensor, float]:
        """

        Args:
            data_x: the data instance to explain of shape (1, num_features)
            label: The label to explain

        Returns:

        """
        if self.mode == "tabular":
            output = self.explainer.explain_instance(data_x[0],
                                                     self.model,
                                                     num_samples=self.n_samples,
                                                     num_features=data_x.shape[1],
                                                     labels=(label,),
                                                     top_labels=None)
            if label not in output.local_exp:
                message = (f"label {label} not in local_explanation! "
                           f"Only labels are {output.local_exp.keys()}")
                raise NameError(message)
            local_explanation = output.local_exp[label]

            # Output requires an array of shape (num_features) where each ith index is the
            # feature importance. We construct this here
            att_arr = [0] * data_x.shape[1]
            for feat_imp in local_explanation:
                att_arr[feat_imp[0]] = feat_imp[1]
            return torch.FloatTensor(att_arr), output.score
        else:
            raise NotImplementedError
