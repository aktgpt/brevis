import json
import os

import brevis.loss as loss
import brevis.model as model

from .data import main as data
from .test import main as test
from .train import main as train


def run(config):
    train_dataloader, valid_dataloader, test_dataloader = data.run(config["data"])
    exp_folder = config["exp_folder"]
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model_save_folder = (
        config["data"]["magnification"]
        + "_"
        + config["data"]["output_channel"]
        + "_"
        + config["train"]["trainer"]
    )

    criterions = []
    for loss_name in config["losses"]:
        model_save_folder += "_" + str(loss_name["weight"]) + "_" + loss_name["type"]
        if loss_name["args"] == None:
            criterions.append(
                {
                    "loss": getattr(loss, loss_name["type"])(),
                    "weight": loss_name["weight"],
                }
            )
        else:
            criterions.append(
                {
                    "loss": getattr(loss, loss_name["type"])(**loss_name["args"]),
                    "weight": loss_name["weight"],
                }
            )

    model_save_folder = os.path.join(exp_folder, model_save_folder)
    save_folder = model_save_folder + "_" + config["train"]["save_name"]
    models = []
    for model_name in config["models"]:
        save_folder += "_" + model_name["type"]
        models.append(getattr(model, model_name["type"])(**model_name["args"]))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, "config.json"), "w") as fp:
        json.dump(config, fp, sort_keys=True, indent=4)

    if config["run_mode"] == "train":
        train.run(
            config["train"],
            train_dataloader,
            valid_dataloader,
            models,
            criterions,
            save_folder,
        )

    test.run(config["test"], test_dataloader, models, save_folder)

