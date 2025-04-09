from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from automedts.askl_typing import FEAT_TYPE_TYPE
from automedts.pipeline.components.base import automedtsPreprocessingAlgorithm
from automedts.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class Densifier(automedtsPreprocessingAlgorithm):
    def __init__(self, random_state=None):
        pass

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        from scipy import sparse

        if sparse.issparse(X):
            return X.todense().getA()
        else:
            return X

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "RandomTreesEmbedding",
            "name": "Random Trees Embedding",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, UNSIGNED_DATA),
            "output": (DENSE, INPUT),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        return cs
