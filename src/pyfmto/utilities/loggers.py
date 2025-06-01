# Reference:
#   https://www.bilibili.com/video/BV1sK4y1x7e1
#   https://www.cnblogs.com/kangshuaibo/p/14700833.html

import logging.config
import logging.handlers
import shutil
from pathlib import Path

__all__ = ['logger', 'reset_log']

LOG_HEAD= r"""
                               ____                __         
            ____     __  __   / __/  ____ ___     / /_   ____ 
           / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
          / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
         / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
        /_/       /____/                                      

"""
LOG_PATH = Path.cwd() / 'out' / 'logs'
LOG_FILE = LOG_PATH / 'pyfmto.log'
LOG_BACKUP = LOG_PATH / 'backup.log'
LOG_CONF = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simpleFormatter': {
            'format': '%(levelname)-8s%(asctime)-22s%(filename)16s->line(%(lineno)s)|%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'pyfmto_handler': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'simpleFormatter',
            'filename': str(LOG_FILE)
        }
    },
    'loggers': {
        'pyfmto': {
            'level': 'INFO',
            'handlers': ['pyfmto_handler'],
            'propagate': 0
        }
    }
}


def _check_path():
    if not LOG_PATH.exists():
        LOG_PATH.mkdir(parents=True)


def _check_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(LOG_HEAD)


def _init_conf():
    logging.config.dictConfig(LOG_CONF)


def reset_log():
    if LOG_FILE.exists():
        shutil.copy(str(LOG_FILE), str(LOG_BACKUP))
        LOG_FILE.unlink()
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(LOG_HEAD)


_check_path()
_check_file()
_init_conf()

logger = logging.getLogger('pyfmto')
