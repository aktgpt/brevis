import argparse
import json
import os
import random

import brevis.exploratory_analysis as exp_analysis
import brevis.main as main
import numpy as np
import torch

os.environ["PYTHONHASHSEED"] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=True
import matplotlib

matplotlib.use("Agg")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="AI-HASTE config file path")
    argparser.add_argument(
        "-c",
        "--conf",
        help="path to configuration file",
    )
    args = argparser.parse_args()
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_device"]
    main.run(config)
    # exp_analysis.run(config)


