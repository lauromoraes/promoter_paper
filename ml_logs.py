#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
@ide: PyCharm
@author: Lauro Ângelo Gonçalves de Moraes
@contact: lauromoraes@ufop.edu.br
@created: 16/06/2020
"""
import mlflow
import logging


class MlLogs(object):
    def __init__(self, verbose=False):
        self._params = dict()
        self._artifacts = dict()
        self._metrics = dict()
        self._verbose = verbose
        logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        if self._verbose:
            msg = 'New MlLogs object create.'
            self.logger.info(msg)

    def set_params(self, **params):
        for k, v in params.items():
            if self._verbose:
                self.logger.debug('Param: {} =\t{}'.format(k, v))
            self._params[k] = v

    def set_atifacts(self, **artifacts):
        for k, v in artifacts.items():
            if self._verbose:
                self.logger.debug('Artifact: {} =\t{}'.format(k, v))
            self._atifacts[k] = v

    def set_metrics(self, **metrics):
        for k, v in metrics.items():
            if self._verbose:
                self.logger.debug('Metric: {} =\t{}'.format(k, v))
            self._metrics[k] = v

    def clear_session(self):
        self._params = dict()
        self._artifacts = dict()
        self._metrics = dict()

    def flush_session(self, run_name):
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metrics(self._metrics)
            mlflow.log_params(self._params)
            mlflow.log_artifacts(self._artifacts)


def main():
    logs = MlLogs(verbose=True)
    logs.set_params(
        conv1_filters=64,
        conv1_k_size=3,
        pool1=3
    )
    logs.set_metrics(
        mean_acc=.901,
        cv0101_acc=.901,
        cv0102_acc=.901,
        cv0201_acc=.901,
        cv0202_acc=.901,
    )
    run_name = ''


if __name__ == "__main__":
    main()
