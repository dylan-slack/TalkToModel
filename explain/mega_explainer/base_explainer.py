"""The base explanation class"""
import torch
import torch.nn as nn


class BaseExplainer(nn.Module):
    """The base explanation class."""

    def __init__(self, model):
        """Init.

        Args:
            model: The model to explain.
        """
        super(BaseExplainer, self).__init__()
        self.model = model

    @staticmethod
    def get_local_neighborhood(x: torch.Tensor) -> torch.Tensor:
        """
            Input : x : any given sample
            This function generates local neighbors of the input sample.
        """

    def get_explanation(self, data_x: torch.FloatTensor, label) -> torch.FloatTensor:
        """
        Input : x : Input sample
        Output : This function uses the explanation model to return explanations for the given sample
        TODO : 1. If we are using public implementations of the explanation methods, do we require predict function?
        """

    def evaluate_explanation(self, explanation: torch.Tensor, evaluation_metric: str,
                             ground_truth: torch.Tensor = None) -> torch.Tensor:
        """
        Input :-
        x : Input explanation for evaluation
        evaluation_metric : the evaluation metric to compute
        ground_truth : expected explanation
        Output : return evaluation metric value.
        """
