from brevis import train as train


def run(config, train_dataloader, valid_dataloader, models, criterions, save_folder):
    trainer = getattr(train, config["trainer"])(config, save_folder)
    save_folder = trainer.train(train_dataloader, valid_dataloader, models, criterions)
    return save_folder
