from __future__ import annotations

import glob
import os
import pickle
import re
import shutil
import tempfile
import time
import uuid
import warnings
from typing import Dict, List, Optional, Tuple, TypeVar, Union, cast
from pathlib import Path

import numpy as np

import pandas as pd

import scipy.sparse

from sklearn.pipeline import Pipeline


from .logging_ import PicklableClientLogger, get_named_client_logger
from ..ensemble_building.abstract_ensemble import AbstractEnsemble



__all__ = ["Backend"]


DATAMANAGER_TYPE = TypeVar("DATAMANAGER_TYPE")
PIPELINE_IDENTIFIER_TYPE = Tuple[int, int, float]


def create(
    temporary_directory: str,
    output_directory: Optional[str],
    prefix: str,
    delete_tmp_folder_after_terminate: bool = True,
    delete_output_folder_after_terminate: bool = True,
) -> "Backend":
    context = BackendContext(
        temporary_directory,
        output_directory,
        delete_tmp_folder_after_terminate,
        delete_output_folder_after_terminate,
        prefix=prefix,
    )
    backend = Backend(context, prefix)

    return backend


def get_randomized_directory_name(
    prefix: str,
    temporary_directory: Optional[str] = None,
) -> str:
    uuid_str = str(uuid.uuid1(clock_seq=os.getpid()))

    temporary_directory = (
        temporary_directory
        if temporary_directory
        else os.path.join(
            tempfile.gettempdir(),
            "{}_tmp_{}".format(
                prefix,
                uuid_str,
            ),
        )
    )

    return temporary_directory


class BackendContext(object):
    def __init__(
        self,
        temporary_directory: str,
        output_directory: Optional[str],
        delete_tmp_folder_after_terminate: bool,
        delete_output_folder_after_terminate: bool,
        prefix: str,
    ):

        # Check that the names of tmp_dir and output_dir is not the same.
        if temporary_directory == output_directory and temporary_directory is not None:
            raise ValueError("The temporary and the output directory " "must be different.")

        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        # attributes to check that directories were properly created
        self._tmp_dir_created = False
        self._output_dir_created = False
        self._prefix = prefix

        self._temporary_directory = get_randomized_directory_name(
            temporary_directory=temporary_directory,
            prefix=self._prefix,
        )
        self._output_directory = output_directory
        # Logging happens via PicklableClientLogger
        # For this reason we need a port to communicate with the server
        # When the backend is created, this port is not available
        # When the port is available in the main process, we
        # call the setup_logger with this port and update self.logger
        self._logger = None  # type: Optional[PicklableClientLogger]
        self.create_directories()

    def setup_logger(self, port: int) -> None:
        self._logger = get_named_client_logger(
            name=__name__,
            port=port,
        )

    @property
    def output_directory(self) -> Optional[str]:
        if self._output_directory is not None:
            # make sure that tilde does not appear on the path.
            return os.path.expanduser(os.path.expandvars(self._output_directory))
        else:
            return None

    @property
    def temporary_directory(self) -> str:
        # make sure that tilde does not appear on the path.
        return os.path.expanduser(os.path.expandvars(self._temporary_directory))

    def create_directories(self, exist_ok: bool = False) -> None:
        # No Exception is raised if self.temporary_directory already exists.
        # This allows to continue the search, also, an error will be raised if
        # save_start_time is called which ensures two searches are not performed
        os.makedirs(self.temporary_directory, exist_ok=exist_ok)
        self._tmp_dir_created = True

        if self.output_directory is not None:
            os.makedirs(self.output_directory, exist_ok=exist_ok)
            self._output_dir_created = True

    def delete_directories(self, force: bool = True) -> None:
        if self.output_directory and (self.delete_output_folder_after_terminate or force):
            if self._output_dir_created is False:
                raise ValueError(
                    "Failed to delete output dir: %s"
                    "Please make sure that the specified output dir did not "
                    "previously exist." % self.output_directory
                )
            try:
                shutil.rmtree(self.output_directory)
            except Exception:
                try:
                    if self._logger is not None:
                        self._logger.warning(
                            "Could not delete output dir: %s" % self.output_directory
                        )
                    else:
                        print("Could not delete output dir: %s" % self.output_directory)
                except Exception:
                    print("Could not delete output dir: %s" % self.output_directory)

        if self.delete_tmp_folder_after_terminate or force:
            if self._tmp_dir_created is False:
                raise ValueError(
                    f"Failed to delete tmp dir {self.temporary_directory}."
                    "Please make sure that the specified tmp dir did not "
                    "previously exist."
                )
            try:
                shutil.rmtree(self.temporary_directory)
            except Exception:
                try:
                    if self._logger is not None:
                        self._logger.warning(
                            "Could not delete tmp dir: %s" % self.temporary_directory
                        )
                    else:
                        print("Could not delete tmp dir: %s" % self.temporary_directory)
                except Exception:
                    print("Could not delete tmp dir: %s" % self.temporary_directory)


class Backend(object):
    """Utility class to load and save all objects to be persisted.

    These are:
    * start time
    * true targets of the ensemble
    """

    def __init__(self, context: BackendContext, prefix: str):
        # When the backend is created, this port is not available
        # When the port is available in the main process, we
        # call the setup_logger with this port and update self.logger
        self.logger = None  # type: Optional[PicklableClientLogger]
        self.context = context
        self.prefix = prefix
        # Track the number of configurations launched
        # num_run == 1 means a dummy estimator run
        self.active_num_run = 1

        # Create the temporary directory if it does not yet exist
        try:
            os.makedirs(self.temporary_directory)
        except Exception:
            pass
        # This does not have to exist or be specified
        if self.output_directory is not None:
            if not os.path.exists(self.output_directory):
                raise ValueError("Output directory %s does not exist." % self.output_directory)

        self.internals_directory = os.path.join(self.temporary_directory, f".{self.prefix}")
        self._make_internals_directory()

    def setup_logger(self, port: int) -> None:
        self.logger = get_named_client_logger(
            name=__name__,
            port=port,
        )
        self.context.setup_logger(port)

    @property
    def output_directory(self) -> Optional[str]:
        return self.context.output_directory

    @property
    def temporary_directory(self) -> str:
        return self.context.temporary_directory

    def _make_internals_directory(self) -> None:
        try:
            os.makedirs(self.internals_directory)
        except Exception as e:
            if self.logger is not None:
                self.logger.debug("_make_internals_directory: %s" % e)
        try:
            os.makedirs(self.get_runs_directory())
        except Exception as e:
            if self.logger is not None:
                self.logger.debug("_make_internals_directory: %s" % e)

    def _get_start_time_filename(self, seed: Union[str, int]) -> str:
        if isinstance(seed, str):
            seed = int(seed)
        return os.path.join(self.internals_directory, "start_time_%d" % seed)

    def save_start_time(self, seed: str) -> str:
        self._make_internals_directory()
        start_time = time.time()

        filepath = self._get_start_time_filename(seed)

        if not isinstance(start_time, float):
            raise ValueError("Start time must be a float, but is %s." % type(start_time))

        if os.path.exists(filepath):
            raise ValueError(
                f"{filepath} already exist. Different seeds should be provided for different jobs."
            )

        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(filepath), delete=False) as fh:
            fh.write(str(start_time))
            tempname = fh.name
        os.rename(tempname, filepath)

        return filepath

    def load_start_time(self, seed: int) -> float:
        with open(self._get_start_time_filename(seed), "r") as fh:
            start_time = float(fh.read())
        return start_time

    def get_smac_output_directory(self) -> str:
        return os.path.join(self.temporary_directory, "smac3-output")

    def get_smac_output_directory_for_run(self, seed: int) -> str:
        return os.path.join(self.temporary_directory, "smac3-output", "run_%d" % seed)

    def _get_targets_ensemble_filename(self, end: str | None = None) -> str:
        dir = Path(self.internals_directory)
        stem = "true_targets_ensemble"
        existing = [p for p in dir.iterdir() if stem in p.name]

        # Sanity check to make sure we have one or None
        assert len(existing) in [0, 1]

        if not any(existing):
            end = "npy"
        else:
            end = existing[0].name.split('.')[-1]

        return os.path.join(self.internals_directory, f"{stem}.{end}")

    def _get_input_ensemble_filename(self, end: str | None = None) -> str:
        dir = Path(self.internals_directory)
        stem = "true_input_ensemble"
        existing = [p for p in dir.iterdir() if stem in p.name]

        # Sanity check to make sure we have one or None
        assert len(existing) in [0, 1]

        if not any(existing):
            end = "npy"
        else:
            end = existing[0].name.split('.')[-1]

        return os.path.join(self.internals_directory, f"{stem}.{end}")

    def save_additional_data(
        self,
        data: Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix],
        what: str,
        overwrite: bool = False,
    ) -> str:
        self._make_internals_directory()
        if isinstance(data, np.ndarray):
            end = "npy"
        elif isinstance(data, scipy.sparse.spmatrix):
            end = "npz"
        elif isinstance(data, pd.DataFrame):
            end = "pd"
        else:
            raise ValueError(
                "Targets must be of type np.ndarray, pd.Dataframe or"
                " scipy.sparse.spmatrix but is %s" % type(data)
            )

        if what == "targets_ensemble":
            filepath = self._get_targets_ensemble_filename(end=end)
        elif what == "input_ensemble":
            filepath = self._get_input_ensemble_filename(end=end)
        else:
            raise ValueError(f"Unknown data type {what}")

        # Only store data if it does not exist yet
        if not overwrite and os.path.isfile(filepath):
            return filepath

        tempname = self._save_array(data=data, filepath=filepath)
        os.rename(tempname, filepath)

        return filepath

    @staticmethod
    def _save_array(
        data: Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix], filepath: str
    ) -> str:
        if isinstance(data, np.ndarray):
            with tempfile.NamedTemporaryFile(
                "wb", dir=os.path.dirname(filepath), delete=False
            ) as fh_w:
                np.save(fh_w, data.astype(np.float32))
        elif isinstance(data, scipy.sparse.spmatrix):
            with tempfile.NamedTemporaryFile(
                "wb", dir=os.path.dirname(filepath), delete=False
            ) as fh_w:
                scipy.sparse.save_npz(fh_w, data)
        elif isinstance(data, pd.DataFrame):
            with tempfile.NamedTemporaryFile(
                "wb", dir=os.path.dirname(filepath), delete=False
            ) as fh_w:
                data.to_pickle(fh_w)
        return fh_w.name

    @staticmethod
    def _load_array(filepath: str) -> np.array:
        end = filepath.split(".")[-1]
        if end == "npy":
            targets = np.load(filepath, allow_pickle=True)
        elif end == "npz":
            targets = scipy.sparse.load_npz(filepath)
        elif end == "pd":
            targets = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unknown file type {end} in {filepath}")

        return targets

    def load_targets_ensemble(self) -> np.ndarray:
        return self._load_array(filepath=self._get_targets_ensemble_filename())

    def load_input_ensemble(self) -> np.ndarray:
        return self._load_array(filepath=self._get_input_ensemble_filename())

    def _get_datamanager_pickle_filename(self) -> str:
        return os.path.join(self.internals_directory, "datamanager.pkl")

    def save_datamanager(self, datamanager: DATAMANAGER_TYPE) -> str:
        self._make_internals_directory()
        filepath = self._get_datamanager_pickle_filename()

        with tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(filepath), delete=False) as fh:
            pickle.dump(datamanager, fh, -1)
            tempname = fh.name
        os.rename(tempname, filepath)

        return filepath

    def load_datamanager(self) -> DATAMANAGER_TYPE:
        filepath = self._get_datamanager_pickle_filename()
        with open(filepath, "rb") as fh:
            return cast(DATAMANAGER_TYPE, pickle.load(fh))

    def get_runs_directory(self) -> str:
        return os.path.join(self.internals_directory, "runs")

    def get_numrun_directory(self, seed: int, num_run: int, budget: float) -> str:
        return os.path.join(self.internals_directory, "runs", "%d_%d_%s" % (seed, num_run, budget))

    def get_next_num_run(self, peek: bool = False) -> int:
        """
        Every pipeline that is fitted by the estimator is stored with an
        identifier called num_run. A dummy classifier will always have a num_run
        equal to 1, and all other new configurations that are explored will
        have a sequentially increasing identifier.

        This method returns the next num_run a configuration should take.

        Parameters
        ----------
        peek: bool
            By default, the next num_rum will be returned, i.e. self.active_num_run + 1
            Yet, if this bool parameter is equal to True, the value of the current
            num_run is provided, i.e, self.active_num_run.
            In other words, peek allows to get the current maximum identifier
            of a configuration.

        Returns
        -------
        num_run: int
            An unique identifier for a configuration
        """

        # If there are other num_runs, their name would be runs/<seed>_<num_run>_<budget>
        other_num_runs = [
            int(os.path.basename(run_dir).split("_")[1])
            for run_dir in glob.glob(os.path.join(self.internals_directory, "runs", "*"))
            if self._is_run_dir(os.path.basename(run_dir))
        ]
        if len(other_num_runs) > 0:
            # We track the number of runs from two forefronts:
            # The physically available num_runs (which might be deleted or a crash could happen)
            # From a internally kept attribute. The later should be sufficient, but we
            # want to be robust against multiple backend copies on different workers
            self.active_num_run = max([self.active_num_run] + other_num_runs)

        # We are interested in the next run id
        if not peek:
            self.active_num_run += 1
        return self.active_num_run

    @staticmethod
    def _is_run_dir(run_dir: str) -> bool:
        """
        Run directories are stored in the format <seed>_<num_run>_<budget>.

        Parameters
        ----------
        run_dir: str
            string containing the base name of the run directory

        Returns
        -------
        _: bool
            whether the provided run directory matches the run_dir_pattern
            signifying that it is a run directory
        """
        run_dir_pattern = r"\d+_\d+_\d+"
        return bool(re.match(run_dir_pattern, run_dir))

    def get_model_filename(self, seed: int, idx: int, budget: float) -> str:
        return "%s.%s.%s.model" % (seed, idx, budget)

    def get_cv_model_filename(self, seed: int, idx: int, budget: float) -> str:
        return "%s.%s.%s.cv_model" % (seed, idx, budget)

    def list_all_models(self, seed: int) -> List[str]:
        runs_directory = self.get_runs_directory()
        model_files = glob.glob(
            os.path.join(glob.escape(runs_directory), "%d_*" % seed, "%s.*.*.model" % seed)
        )
        return model_files

    def load_models_by_identifiers(
        self, identifiers: List[PIPELINE_IDENTIFIER_TYPE]
    ) -> Dict[PIPELINE_IDENTIFIER_TYPE, Pipeline]:
        models = {}

        for identifier in identifiers:
            seed, idx, budget = identifier
            models[identifier] = self.load_model_by_seed_and_id_and_budget(seed, idx, budget)

        return models

    def load_model_by_seed_and_id_and_budget(self, seed: int, idx: int, budget: float) -> Pipeline:
        model_directory = self.get_numrun_directory(seed, idx, budget)

        model_file_name = "%s.%s.%s.model" % (seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)
        with open(model_file_path, "rb") as fh:
            return pickle.load(fh)

    def load_cv_models_by_identifiers(
        self, identifiers: List[PIPELINE_IDENTIFIER_TYPE]
    ) -> Dict[PIPELINE_IDENTIFIER_TYPE, Pipeline]:
        models = {}

        for identifier in identifiers:
            seed, idx, budget = identifier
            models[identifier] = self.load_cv_model_by_seed_and_id_and_budget(seed, idx, budget)

        return models

    def load_cv_model_by_seed_and_id_and_budget(
        self, seed: int, idx: int, budget: float
    ) -> Pipeline:
        model_directory = self.get_numrun_directory(seed, idx, budget)

        model_file_name = "%s.%s.%s.cv_model" % (seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)
        with open(model_file_path, "rb") as fh:
            return pickle.load(fh)

    def save_numrun_to_dir(
        self,
        seed: int,
        idx: int,
        budget: float,
        model: Optional[Pipeline],
        cv_model: Optional[Pipeline],
        ensemble_predictions: Optional[np.ndarray],
        valid_predictions: Optional[np.ndarray],
        test_predictions: Optional[np.ndarray],
    ) -> None:
        runs_directory = self.get_runs_directory()
        tmpdir = tempfile.mkdtemp(dir=runs_directory)
        if model is not None:
            file_path = os.path.join(tmpdir, self.get_model_filename(seed, idx, budget))
            with open(file_path, "wb") as fh:
                pickle.dump(model, fh, -1)

        if cv_model is not None:
            file_path = os.path.join(tmpdir, self.get_cv_model_filename(seed, idx, budget))
            with open(file_path, "wb") as fh:
                pickle.dump(cv_model, fh, -1)

        for preds, subset in (
            (ensemble_predictions, "ensemble"),
            (valid_predictions, "valid"),
            (test_predictions, "test"),
        ):
            if preds is not None:
                file_path = os.path.join(
                    tmpdir, self.get_prediction_filename(subset, seed, idx, budget)
                )
                with open(file_path, "wb") as fh:
                    pickle.dump(preds.astype(np.float32), fh, -1)
        try:
            os.rename(tmpdir, self.get_numrun_directory(seed, idx, budget))
        except OSError:
            if os.path.exists(self.get_numrun_directory(seed, idx, budget)):
                os.rename(
                    self.get_numrun_directory(seed, idx, budget),
                    os.path.join(runs_directory, tmpdir + ".old"),
                )
                os.rename(tmpdir, self.get_numrun_directory(seed, idx, budget))
                shutil.rmtree(os.path.join(runs_directory, tmpdir + ".old"))

    def get_ensemble_dir(self) -> str:
        return os.path.join(self.internals_directory, "ensembles")

    def load_ensemble(self, seed: int) -> Optional[AbstractEnsemble]:
        ensemble_dir = self.get_ensemble_dir()

        if not os.path.exists(ensemble_dir):
            if self.logger is not None:
                self.logger.warning("Directory %s does not exist" % ensemble_dir)
            else:
                warnings.warn("Directory %s does not exist" % ensemble_dir)
            return None

        if seed >= 0:
            indices_files = glob.glob(
                os.path.join(glob.escape(ensemble_dir), "%s.*.ensemble" % seed)
            )
            indices_files.sort()
        else:
            indices_files = os.listdir(ensemble_dir)
            indices_files = [os.path.join(ensemble_dir, f) for f in indices_files]
            indices_files.sort(key=lambda f: time.ctime(os.path.getmtime(f)))

        with open(indices_files[-1], "rb") as fh:
            ensemble_members_run_numbers = cast(AbstractEnsemble, pickle.load(fh))

        return ensemble_members_run_numbers

    def save_ensemble(self, ensemble: AbstractEnsemble, idx: int, seed: int) -> None:
        try:
            os.makedirs(self.get_ensemble_dir())
        except Exception:
            pass

        filepath = os.path.join(
            self.get_ensemble_dir(), "%s.%s.ensemble" % (str(seed), str(idx).zfill(10))
        )
        with tempfile.NamedTemporaryFile("wb", dir=os.path.dirname(filepath), delete=False) as fh:
            pickle.dump(ensemble, fh)
            tempname = fh.name
        os.rename(tempname, filepath)

    def get_prediction_filename(
        self, subset: str, automl_seed: Union[str, int], idx: int, budget: float
    ) -> str:
        return "predictions_%s_%s_%s_%s.npy" % (subset, automl_seed, idx, budget)

    def save_predictions_as_txt(
        self,
        predictions: np.ndarray,
        subset: str,
        idx: int,
        precision: int,
        prefix: Optional[str] = None,
    ) -> None:
        if not self.output_directory:
            return
        # Write prediction scores in prescribed format
        filepath = os.path.join(
            self.output_directory,
            ("%s_" % prefix if prefix else "") + "%s_%s.predict" % (subset, str(idx)),
        )

        format_string = "{:.%dg} " % precision
        with tempfile.NamedTemporaryFile(
            "w", dir=os.path.dirname(filepath), delete=False
        ) as output_file:
            for row in predictions:
                if not isinstance(row, np.ndarray) and not isinstance(row, list):
                    row = [row]
                for val in row:
                    output_file.write(format_string.format(float(val)))
                output_file.write("\n")
            tempname = output_file.name
        os.rename(tempname, filepath)

    def write_txt_file(self, filepath: str, data: str, name: str) -> None:
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(filepath), delete=False) as fh:
            fh.write(data)
            tempname = fh.name
        os.rename(tempname, filepath)
        if self.logger is not None:
            self.logger.debug("Created %s file %s" % (name, filepath))
