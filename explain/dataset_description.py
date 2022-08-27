"""A class that stores dataset specific descriptions specified for that dataset.

These descriptions are called in different ways when generating responses in the conversation to
provide more tailored dataset specified feedback
"""
import gin
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

from explain.utils import read_and_format_data


@gin.configurable
class DatasetDescription:
    """The dataset description class."""

    def __init__(self,
                 dataset_objective: str = "",
                 dataset_description: str = "",
                 model_description: str = "",
                 eval_file_path: str = None,
                 index_col: int = 0,
                 target_var_name: str = "y"):
        """Init.

        Arguments:
            dataset_objective: The goal of the dataset. I.e., "predict whether someone has a
                               disease" or "predict whether this image is a dog".
            dataset_description: A brief description of the dataset, i.e., "disease prediction" or
                                 "canine classification".
            eval_file_path: The filepath to an eval dataset that will be used to compute a test
                            score for the model on the training dataset to summarize performance.
            index_col: The index columns of the testing data
            target_var_name: The target variable name in the testing data
            model_description: A description of the model. i.e., gradient boosted tree or linear
                               regression
        """
        self.objective = dataset_objective
        self.description = dataset_description
        self.eval_file_path = eval_file_path
        self.index_col = index_col
        self.target_var_name = target_var_name
        self.model_description = model_description

    def get_dataset_objective(self):
        """Gets the objective."""
        return self.objective

    def get_dataset_description(self):
        """Gets the description."""
        return self.description

    def get_model_description(self):
        """Gets the model description."""
        return self.model_description

    def get_text_description(self):
        """Returns a brief text overview of the dataset and model."""
        text = (f"This chat interfaces to a model trained on a {self.get_dataset_description()}"
                f" dataset. The goal of the model is to {self.get_dataset_objective()}.")
        return text

    @staticmethod
    def get_score_text(y_true: Any,
                       y_pred: Any,
                       metric_name: str,
                       rounding_precision: int,
                       data_name: str) -> str:
        """Computes model score and returns text describing the outcome.

        Arguments:
            data_name: The name of the data split, e.g. testing data
            y_true: The true y values
            y_pred: The predicted y values
            metric_name: The name of the metric
            rounding_precision: The sig figs to round to
        Returns:
            performance_summary: A string describing the performance
        """
        if metric_name == "accuracy":
            score = accuracy_score(y_true, y_pred)
            # sklearn defaults to accuracy represented as decimal. convert this to %
            score *= 100
        elif metric_name == "roc":
            score = roc_auc_score(y_true, y_pred)
        elif metric_name == "f1":
            score = f1_score(y_true, y_pred)
        elif metric_name == "recall":
            score = recall_score(y_true, y_pred)
        elif metric_name == "precision":
            score = precision_score(y_true, y_pred)
        elif metric_name == "sensitivity":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = tp / (tp + fn)
        elif metric_name == "specificity":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = tn / (tn + fp)
        elif metric_name == "ppv":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = tp / (tp + fp)
        elif metric_name == "npv":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = tn / (tn + fn)
        else:
            raise NameError(f"Unknown metric {metric_name}")

        string_score = str(round(score, rounding_precision))

        # additional context for accuracy score
        if metric_name == "accuracy":
            string_score += "%"

        performance_summary = f"The model scores <em>{string_score} {metric_name}</em> on "
        performance_summary += f"{data_name}."
        return performance_summary

    def get_eval_performance(self,
                             model: Any,
                             metric_name: str = "accuracy",
                             rounding_precision: int = 3) -> str:
        """Computes the eval performance.

        Arguments:
            model: The model
            metric_name: The name of the metric used, e.g., accuracy. The currently supported
                         metrics are accuracy, roc, f1, recall, and precision.
            rounding_precision: The number of decimal places to present in the result
        Returns:
            performance_summary: A string describing the performance summary of the model.
        """

        # If no eval dataset is specified, ignore providing
        # performance summary
        if self.eval_file_path is None:
            return ""

        # Loads and processes the testing dataset
        x_values, y_values, _, _ = read_and_format_data(self.eval_file_path,
                                                        index_col=self.index_col,
                                                        target_var_name=self.target_var_name,
                                                        cat_features=None,
                                                        num_features=None)

        # read_and_format_data returns pandas.df, so convert to numpy for model inference
        x_values = x_values.values
        y_pred = model.predict(x_values)
        # Get performance summary
        performance_summary = self.get_score_text(y_values,
                                                  y_pred,
                                                  metric_name,
                                                  rounding_precision,
                                                  "the data")
        return performance_summary
