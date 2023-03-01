import os
from typing import List
import pickle as pkl
import gin
import numpy as np


def load_cache(cache_location: str):
    """Loads the cache."""
    if os.path.isfile(cache_location):
        with open(cache_location, 'rb') as file:
            cache = pkl.load(file)
    else:
        cache = []
    return cache


@gin.configurable
class DiverseInstances:
    """This class finds DiverseInstances by using LIMEs submodular pick."""

    def __init__(self,
                 cache_location: str = "./cache/diverse-instances.pkl",
                 lime_explainer=None):
        """

        Args:
            cache_location: location to save the cache
            lime_explainer: lime explainer to use for finding diverse instances (from MegaExplainer)
        """
        self.diverse_instances = load_cache(cache_location)
        self.cache_location = cache_location
        self.lime_explainer = lime_explainer

    def get_diverse_instances(self,
                              data: np.ndarray,
                              instance_count: int = 10,
                              save_to_cache=True) -> List[int]:
        """
        Returns diverse instances for the given data set.
        Args:
            data: the data instance to explain of shape (num_samples, num_features)
            instance_count: number of diverse instances to return
            save_to_cache: whether to save the diverse instances to the cache
        Returns: List of diverse instance ids.

        """
        """Uses LIME explainer to find diverse instances by submodular pick."""
        if len(self.diverse_instances) > 0:
            return self.diverse_instances

        # Generate diverse instances
        diverse_instances = self.lime_explainer.get_diverse_instances(data[:300].values, instance_count)
        if save_to_cache:
            with open(self.cache_location, 'wb') as file:
                pkl.dump(diverse_instances, file)
        return diverse_instances
