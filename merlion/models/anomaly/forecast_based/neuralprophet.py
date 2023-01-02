#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Adaptation of NeuralProphet forecasting model to anomaly detection.
"""

from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.anomaly.base import DetectorConfig
from merlion.models.forecast.NeuralProphet import NeuralProphetConfig, NeuralProphet
from merlion.post_process.threshold import AggregateAlarms


class NeuralProphetDetectorConfig(NeuralProphetConfig, DetectorConfig):
    _default_threshold = AggregateAlarms(alm_threshold=3)


class NeuralProphetDetector(ForecastingDetectorBase, NeuralProphet):
    config_class = NeuralProphetDetectorConfig
