import logging 


handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

