import random
import os
import logging

def setup_file_logger(logname):
    logging.shutdown()
    handler = logging.FileHandler(logname)
    formatter = logging.Formatter('%(asctime)s-%(message)s', "%m-%d-%H:%M:%S")
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logger.addHandler(handler)
    return logger

def delete_if_exist(fn):
    if os.path.exists(fn):
        os.remove(fn)
    return


def maybe_create(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def polyak_update(polyak_factor, target_network, network):  
    params1 = network.state_dict()
    params2 = target_network.state_dict()

    for name1, param1 in params1.items():
        if name1 in params2:
            params2[name1].data.copy_(polyak_factor * param1.data + (1 - polyak_factor) * params2[name1].data)

    #model.load_state_dict(dict_params2)

    target_network.load_state_dict(params2)