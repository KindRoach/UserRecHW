import logging


def get_logger(name: str = "logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log/' + name + '.log')
    # fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger if logger is newly created.
    if not logger.handlers:
        # logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


logger = get_logger("UserRecHW")
