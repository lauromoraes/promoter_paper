#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@ide: PyCharm
@author: Lauro Ângelo Gonçalves de Moraes
@contact: lauromoraes@ufop.edu.br
@created: 16/06/2020
"""
import os
import mlflow
import logging
from mlflow.tracking import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id
import numpy as np


class MlLogs(object):
    def __init__(self, name, verbose=False, artifacts_folder='.'):
        self._name = name
        self._params = dict()
        self._artifacts = list()
        self._metrics = dict()
        self._verbose = verbose
        self._client = MlflowClient()
        self._n_partitions = None
        self._n_samples = None
        self._tags = {}
        self._imgs_path = None
        self._model_path = None
        self._artifacts_folders = None
        self.create_artifacts_folders(artifacts_folder)
        self._actual_experiment_id = _get_experiment_id()
        logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        if self._verbose:
            msg = 'New MlLogs object create.'
            self.logger.info(msg)

    def init_stats(self, metrics_names, n_partitions, n_samples):
        self._n_partitions = n_partitions
        self._n_samples = n_samples
        for _p_idx in range(n_partitions):
            for _metric_name in metrics_names:
                print(_metric_name)

    def set_tag(self, key, value):
        self._tags[key] = value

    def set_stats(self, partion_idx, stats):
        for _metric_name in stats.get_metrics_types():
            prefix = 'p{:02d}_{}'.format(partion_idx, _metric_name)

    def create_artifacts_folders(self, root_path=None):
        base_path = 'objects_{}'.format(self._name)
        base_path = os.path.join(root_path, 'objects', base_path)
        os.makedirs(base_path, mode=0o777, exist_ok=False)
        self._artifacts_folders = base_path

        img_path = os.path.join(base_path, 'imgs')
        os.makedirs(img_path, mode=0o777, exist_ok=False)
        self._imgs_path = img_path

        model_path = os.path.join(base_path, 'model')
        # os.makedirs(model_path, mode=0o777, exist_ok=False)
        self._model_path = model_path

    def set_image(self, img_path):
        self._artifacts.append(img_path)
        return img_path

    def get_image_path(self):
        return self._imgs_path

    def get_model_path(self, partiton):
        return '{}_p{:02d}'.format(self._model_path, partiton)

    def get_artifacts_folder(self):
        return self._artifacts_folders

    def set_params(self, **params):
        for k, v in params.items():
            if self._verbose:
                self.logger.debug('Param: {} =\t{}'.format(k, v))
            self._params[k] = v

    def set_artifact(self, item):
        if self._verbose:
            self.logger.debug('Artifact: {}'.format(item))
        self._artifacts.append(item)

    def set_artifacts(self, **artifacts):
        for i in artifacts:
            if self._verbose:
                self.logger.debug('Artifact: {}'.format(i))
            self._artifacts.append(i)

    def set_metrics(self, **metrics):
        for k, v in metrics.items():
            if self._verbose:
                self.logger.debug('Metric: {} =\t{}'.format(k, v))
            self._metrics[k] = v

    def log_metrics(self, metrics):
        for _m_label, _metric in metrics.items():
            # print(_m_label, _metric)
            if isinstance(_metric, dict):
                n_partitions = len(_metric.keys())
                _mean = np.zeros(n_partitions)
                _std = np.zeros(n_partitions)
                _max = np.zeros(n_partitions)
                _min = np.zeros(n_partitions)
                # Iterate over partitions
                for _p_idx, _p_label in enumerate(_metric.keys()):
                    _samples = _metric[_p_label]
                    if isinstance(_samples, list) or isinstance(_samples, np.ndarray):
                        # Calculate samples stats of partition
                        _mean[_p_idx] = np.average(_samples)
                        _std[_p_idx] = np.std(_samples)
                        _max[_p_idx] = np.amax(_samples)
                        _min[_p_idx] = np.amin(_samples)
                        # Log samples stats of partition
                        mlflow.log_metric(key='{}_mean'.format(_p_label), value=_mean[_p_idx])
                        mlflow.log_metric(key='{}_std'.format(_p_label), value=_std[_p_idx])
                        mlflow.log_metric(key='{}_max'.format(_p_label), value=_max[_p_idx])
                        mlflow.log_metric(key='{}_min'.format(_p_label), value=_min[_p_idx])
                        # Log samples values of partition
                        for _s_idx, sample_metric in enumerate(_samples):
                            mlflow.log_metric(key='{}_samples'.format(_p_label), value=sample_metric, step=_s_idx)
                # Log stats
                mlflow.log_metric(key='{}_avg_mean'.format(_m_label), value=np.average(_mean))
                mlflow.log_metric(key='{}_avg_std'.format(_m_label), value=np.average(_std))
                mlflow.log_metric(key='{}_avg_max'.format(_m_label), value=np.average(_max))
                mlflow.log_metric(key='{}_avg_min'.format(_m_label), value=np.average(_min))
            else:
                mlflow.log_metric(key=_m_label, value=_metric)

    def clear_session(self):
        self._params = dict()
        self._artifacts = dict()
        self._metrics = dict()

    def set_experiment(self, experiment_name):
        # experiments = self._client.list_experiments()  # returns a list of mlflow.entities.Experiment
        experiments = {x.name: x.experiment_id for x in self._client.list_experiments()}
        if experiment_name not in experiments.keys():
            self._client.create_experiment(experiment_name)
            print('Created new experiment.')
        else:
            print('Using previous experiment.')
        experiments = {x.name: x.experiment_id for x in self._client.list_experiments()}
        _id = self._actual_experiment_id = experiments[experiment_name]
        return _id

    def flush_session(self, experiment_name=None, run_name=None):
        mlflow.end_run()
        if experiment_name is not None:
            self.set_experiment(experiment_name=experiment_name)

        # run = self._client.create_run(experiments[0].experiment_id)  # returns mlflow.entities.Run
        # self._client.log_param(run.info.run_id, "hello", "world")
        # self._client.set_terminated(run.info.run_id)

        with mlflow.start_run(experiment_id=self._actual_experiment_id, run_name=run_name):
            self.log_metrics(self._metrics)
            mlflow.log_params(self._params)
            mlflow.set_tags(self._tags)
            mlflow.log_artifacts(self._artifacts_folders)
            mlflow.end_run()


class MyExperiment(object):
    def __init__(self, exp_name, num_partitions):
        self._exp_name = exp_name
        self._num_partitions = num_partitions
        self._partitions = [dict() for _ in range(self.num_partitions)]

    def set_partition_sample_stats(self, idx_partition, sample_stats):
        d = sample_stats.to_dict()
        for k, v in d.items():
            self._partitions[idx_partition][k] = v


def main():
    from ml_statistics import BaseStatistics

    # ==== STATS - generate synthetic statistics
    def gen_synthetic_stats(num_samples=100, similarity=.8):
        y_true = np.random.randint(2, size=num_samples)
        y_pred = np.array([x if np.random.random() < similarity else 1 - x for x in y_true])
        statistics = BaseStatistics(y_true, y_pred)
        print(statistics)
        return statistics

    # ==== STATS - define model stats object
    stats = gen_synthetic_stats(num_samples=100, similarity=.8)
    print(stats.to_dict())

    # ==== LOGS - experiment and run selection
    model_type = 'HOT_CNN_Bacillus'
    # ==== LOGS - setting log values
    experiment_name = 'SampleExecutions'
    run_name = '{}'.format(model_type)
    logs = MlLogs(verbose=True)
    logs.set_params(
        initial_lr=1e-3,
        lr_decay=.9,
        batch_size=16,
    )
    n_partitions = 5
    n_samples = 10
    partition_idx = 0
    sample_idx = 0

    acc = {'p{:02d}_acc'.format(x): np.random.normal(loc=.9, scale=.02, size=n_samples) for x in range(n_partitions)}
    mcc = {'p{:02d}_mcc'.format(x): np.random.normal(loc=.82, scale=.03, size=n_samples) for x in range(n_partitions)}
    logs.set_metrics(
        acc=acc,
        mcc=mcc,
    )
    # ==== LOGS - send outputs
    logs.flush_session(experiment_name=experiment_name, run_name=run_name)
    logs.clear_session()


if __name__ == "__main__":
    main()
