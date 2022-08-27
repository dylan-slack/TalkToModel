"""TODO(satya): docstring."""
import torch


class BasePerturbation:
    """Base Class for perturbation methods."""

    def __init__(self, data_format):
        """Initialize generic parameters for the perturbation method."""
        assert data_format == "tabular", "Currently, only tabular data is supported!"
        self.data_format = data_format

    def get_perturbed_inputs(self,
                             original_sample: torch.FloatTensor,
                             feature_mask: torch.BoolTensor,
                             num_samples: int,
                             feature_metadata: list,
                             max_distance: int = None) -> torch.tensor:
        """Logic of the perturbation methods which will return perturbed samples.

        This method should be overwritten.
        """


class NormalPerturbation(BasePerturbation):
    """TODO(satya): docstring.

    TODO(satya): Should we scale the std. based on the size of the feature? This could lead to
    some odd results if the features aren't scaled the same and we apply the same std noise
    across all the features.
    """

    def __init__(self,
                 data_format,
                 mean: float = 0.0,
                 std: float = 0.05,
                 flip_percentage: float = 0.3):
        """Init.

        Args:
            data_format: A string describing the format of the data, i.e., "tabular" for tabular
                         data.
            mean: the mean of the gaussian perturbations
            std: the standard deviation of the gaussian perturbations
            flip_percentage: The percent of features to flip while perturbing
        """
        self.mean = mean
        self.std_dev = std
        self.flip_percentage = flip_percentage
        super(NormalPerturbation, self).__init__(data_format)

    def get_perturbed_inputs(self,
                             original_sample: torch.FloatTensor,
                             feature_mask: torch.BoolTensor,
                             num_samples: int,
                             feature_metadata: list,
                             max_distance: int = None) -> torch.tensor:
        """Given a sample and mask, compute perturbations.

        Args:
            original_sample: The original instance
            feature_mask: the indices of the indices to mask where True corresponds to an index
                          that is to be masked. E.g., [False, True, False] means that index 1 will
                          not be perturbed while 0 and 2 _will_ be perturbed.
            num_samples: number of perturbed samples.
            feature_metadata: the list of 'c' or 'd' for whether the feature is categorical or
                              discrete.
            max_distance: the maximum distance between original sample and perturbed samples.
        Returns:
            perturbed_samples: The original_original sample perturbed with Gaussian perturbations
                               num_samples times.
        """
        feature_type = feature_metadata

        message = f"mask size == original sample in get_perturbed_inputs for {self.__class__}"
        assert len(feature_mask) == len(original_sample), message

        continuous_features = torch.tensor([i == 'c' for i in feature_type])
        discrete_features = torch.tensor([i == 'd' for i in feature_type])

        # Processing continuous columns
        mean = self.mean
        std_dev = self.std_dev
        perturbations = torch.normal(mean, std_dev,
                                     [num_samples, len(feature_type)]) * continuous_features + original_sample

        # Processing discrete columns
        flip_percentage = self.flip_percentage
        p = torch.empty(num_samples, len(feature_type)).fill_(flip_percentage)
        perturbations = perturbations * (~discrete_features) + torch.abs(
            (perturbations * discrete_features) - (torch.bernoulli(p) * discrete_features))

        # keeping features static that are in top-K based on feature mask
        perturbed_samples = torch.tensor(
            original_sample) * feature_mask + perturbations * (~feature_mask)

        return perturbed_samples
