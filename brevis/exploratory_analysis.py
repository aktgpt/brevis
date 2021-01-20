from .utils.make_split import DataSplitter
from .utils.exp_stats import ExperimentStatsGetter
import os
from .utils.get_nuclei_centroids import CentroidExtractor


def run(config):
    data_folder = config["data"]["folder"]

    exp_stats_folder = "exp_stats"
    if not os.path.exists(exp_stats_folder):
        os.makedirs(exp_stats_folder)
    centroid_extractor = CentroidExtractor(data_folder, exp_stats_folder)
    centroid_extractor.get_centroids()
    # exp_stats_getter = ExperimentStatsGetter(data_folder, exp_stats_folder)
    # exp_stats_getter.get_stats()

