from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union
import random
import warnings
from collections import Counter
import numpy as np
from sklearn.utils import check_random_state

from automedts.automl_common.common.utils.backend import Backend
from automedts.constants import TASK_TYPES
from automedts.data.validation import SUPPORTED_FEAT_TYPES
from automedts.ensemble_building.run import Run
from automedts.ensembles.abstract_ensemble import AbstractEnsemble
from automedts.metrics import Scorer, calculate_losses
from automedts.pipeline.base import BasePipeline


class EnsembleSelection(AbstractEnsemble):
    def __init__(
        self,
        task_type: int,
        metrics: Sequence[Scorer] | Scorer,
        backend: Backend,
        ensemble_size: int = 50,
        bagging: bool = False,
        mode: str = "fast",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        if isinstance(metrics, Sequence):
            if len(metrics) > 1:
                warnings.warn(
                    "Ensemble selection can only optimize one metric, "
                    "but multiple metrics were passed, dropping all "
                    "except for the first metric."
                )
            self.metric = metrics[0]
        else:
            self.metric = metrics
        self.bagging = bagging
        self.mode = mode
        self.random_state = random_state

        # Control the uncertainty penalty weight
        self.uncertainty_penalty = 0.15  # 

    def fit(
        self,
        base_models_predictions: List[np.ndarray],
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
        runs: Sequence[Run],
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> EnsembleSelection:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError("Ensemble size cannot be less than one!")
        if self.task_type not in TASK_TYPES:
            raise ValueError("Unknown task type %s." % self.task_type)
        if not isinstance(self.metric, Scorer):
            raise ValueError("Invalid metric type: %s" % type(self.metric))
        if self.mode not in ("fast", "slow"):
            raise ValueError("Unknown mode %s" % self.mode)

        if self.bagging:
            self._bagging(base_models_predictions, true_targets)
        else:
            self._fit(
                predictions=base_models_predictions,
                X_data=X_data,
                labels=true_targets,
            )
        self._calculate_weights()
        self.identifiers_ = model_identifiers
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        *,
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> EnsembleSelection:
        if self.mode == "fast":
            self._fast(predictions=predictions, X_data=X_data, labels=labels)
        else:
            self._slow(predictions=predictions, X_data=X_data, labels=labels)
        return self

    def _fast(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        *,
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> None:
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size
        weighted_ensemble_prediction = np.zeros(predictions[0].shape, dtype=np.float64)
        fant_ensemble_prediction = np.zeros_like(weighted_ensemble_prediction)

        for i in range(ensemble_size):
            losses = np.zeros(len(predictions), dtype=np.float64)
            s = len(ensemble)

            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            for j, pred in enumerate(predictions):
                np.add(weighted_ensemble_prediction, pred, out=fant_ensemble_prediction)
                np.multiply(fant_ensemble_prediction, 1.0 / (s + 1), out=fant_ensemble_prediction)

                loss = calculate_losses(
                    solution=labels,
                    prediction=fant_ensemble_prediction,
                    task_type=self.task_type,
                    metrics=[self.metric],
                    X_data=X_data,
                    scoring_functions=None,
                )[self.metric.name]

                # Incorporate uncertainty awareness: the larger the variance, the greater the penalty.
                variance_penalty = self.uncertainty_penalty * np.var(pred)
                losses[j] = loss + variance_penalty

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
            best = rand.choice(all_best)

            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_loss_ = trajectory[-1]

    def _slow(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        *,
        X_data: SUPPORTED_FEAT_TYPES | None = None,
    ) -> None:
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size
        for i in range(ensemble_size):
            losses = np.zeros(len(predictions), dtype=np.float64)
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)

                loss = calculate_losses(
                    solution=labels,
                    prediction=ensemble_prediction,
                    task_type=self.task_type,
                    metrics=[self.metric],
                    X_data=X_data,
                    scoring_functions=None,
                )[self.metric.name]

                variance_penalty = self.uncertainty_penalty * np.var(pred)
                losses[j] = loss + variance_penalty
                ensemble.pop()

            best = np.nanargmin(losses)
            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            if len(predictions) == 1:
                break

        self.indices_ = np.array(order, dtype=np.int64)
        self.trajectory_ = np.array(trajectory, dtype=np.float64)
        self.train_loss_ = trajectory[-1]

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(self.num_input_models_, dtype=np.float64)
        for model_id, count in ensemble_members:
            weights[model_id] = count / self.ensemble_size
        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)
        self.weights_ = weights

    def _bagging(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        fraction: float = 0.5,
        n_bags: int = 20,
    ) -> np.ndarray:
        raise ValueError("Bagging might not work with class-based interface!")

    def predict(
        self, base_models_predictions: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        average = np.zeros_like(base_models_predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(base_models_predictions[0], dtype=np.float64)

        if len(base_models_predictions) == len(self.weights_):
            for pred, weight in zip(base_models_predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)
        elif len(base_models_predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(base_models_predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)
        else:
            raise ValueError("Mismatch between predictions and weights")
        del tmp_predictions
        return average

    def __str__(self) -> str:
        trajectory_str = " ".join(
            [f"{i}: {loss:.5f}" for i, loss in enumerate(self.trajectory_)]
        )
        identifiers_str = " ".join(
            f"{self.identifiers_[i]}" for i in range(len(self.weights_)) if self.weights_[i] > 0
        )
        return (
            f"Ensemble Selection:\n"
            f"\tTrajectory: {trajectory_str}\n"
            f"\tMembers: {self.indices_}\n"
            f"\tWeights: {self.weights_}\n"
            f"\tIdentifiers: {identifiers_str}\n"
        )

    def get_models_with_weights(
        self, models: Dict[Tuple[int, int, float], BasePipeline]
    ) -> List[Tuple[float, BasePipeline]]:
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                output.append((weight, models[identifier]))
        output.sort(reverse=True, key=lambda x: x[0])
        return output

    def get_identifiers_with_weights(self) -> List[Tuple[Tuple[int, int, float], float]]:
        return list(zip(self.identifiers_, self.weights_))

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        return [
            self.identifiers_[i]
            for i, weight in enumerate(self.weights_)
            if weight > 0.0
        ]

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]
