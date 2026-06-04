import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "errors.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

def _setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    root_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO); ch.setFormatter(formatter)
    root_logger.addHandler(ch)
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG); fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    except OSError as e:
        root_logger.warning("Could not create app log file: %s", e)
    try:
        eh = RotatingFileHandler(ERROR_LOG_FILE, maxBytes=2*1024*1024, backupCount=3, encoding="utf-8")
        eh.setLevel(logging.ERROR); eh.setFormatter(formatter)
        root_logger.addHandler(eh)
    except OSError as e:
        root_logger.warning("Could not create error log file: %s", e)
    root_logger.info("Logging initialised | app_log=%s | error_log=%s", LOG_FILE, ERROR_LOG_FILE)

_setup_logging()

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)