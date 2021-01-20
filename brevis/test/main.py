from brevis import test as test


def run(config, dataloader, models, save_folder):
    tester = getattr(test, config["tester"])(config, save_folder)
    tester.test(dataloader, models)
