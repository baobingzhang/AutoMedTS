from typing import Dict, Optional, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from automedts.askl_typing import FEAT_TYPE_TYPE
from automedts.pipeline.base import DATASET_PROPERTIES_TYPE, PIPELINE_DATA_DTYPE
from automedts.pipeline.components.base import automedtsPreprocessingAlgorithm
from automedts.pipeline.constants import DENSE, INPUT, SPARSE, UNSIGNED_DATA


class NoEncoding(automedtsPreprocessingAlgorithm):
    def __init__(
        self, random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        pass

    def fit(
        self, X: PIPELINE_DATA_DTYPE, y: Optional[PIPELINE_DATA_DTYPE] = None
    ) -> "NoEncoding":
        self.preprocessor = "passthrough"
        self.fitted_ = True
        return self

    def transform(self, X: PIPELINE_DATA_DTYPE) -> PIPELINE_DATA_DTYPE:
        return X

    @staticmethod
    def get_properties(
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> Dict[str, Optional[Union[str, int, bool, Tuple]]]:
        return {
            "shortname": "no encoding",
            "name": "No categorical variable encoding",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "handles_sparse": True,
            "handles_dense": True,
            "input": (DENSE, SPARSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None,
        dataset_properties: Optional[DATASET_PROPERTIES_TYPE] = None,
    ) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
