import argparse
import json
import os
import random

# import setproctitle
import torch
# import horovod.torch as hvd
import numpy as np
# print ('Available devices ', torch.cuda.device_count())
# hvd.init()
# torch.manual_seed(42)
# torch.cuda.set_device(hvd.local_rank())
# print(
# "size=", hvd.size(),
# "global_rank=", hvd.rank(),
# "local_rank=", hvd.local_rank(),
# "device=", torch.cuda.get_device_name(hvd.local_rank())
# )
import ai_haste.main as main
import ai_haste.exploratory_analysis as exp_analysis
# print ('Available devices ', torch.cuda.device_count())
# setproctitle.setproctitle("4 more years!!")

# os.environ["DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
        # default="configs/config_dual_skip.json",
        help="path to configuration file",
    )
    args = argparser.parse_args()
    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_device"]
    # exp_analysis.run(config)
    main.run(config)

