# Reference:
#   https://www.bilibili.com/video/BV1sK4y1x7e1
#   https://www.cnblogs.com/kangshuaibo/p/14700833.html

import logging.config
import logging.handlers
import shutil
from pathlib import Path

__all__ = ['logger', 'reset_log', 'backup_log_to']

LOG_HEAD= r"""
                               ____                __         
            ____     __  __   / __/  ____ ___     / /_   ____ 
           / __ \   / / / /  / /_   / __ `__ \   / __/  / __ \
          / /_/ /  / /_/ /  / __/  / / / / / /  / /_   / /_/ /
         / .___/   \__, /  /_/    /_/ /_/ /_/   \__/   \____/ 
        /_/       /____/                                      

"""
LOG_PATH = Path('out', 'logs')
LOG_PATH.mkdir(parents=True, exist_ok=True)
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


def _init_file():
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        shutil.copy(LOG_FILE, LOG_BACKUP)
    with LOG_FILE.open('w', encoding='utf-8') as f:
        f.write(LOG_HEAD)

def backup_log_to(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dist_file = dest_dir / LOG_BACKUP.name
    if LOG_BACKUP.exists() and not dist_file.exists():
        shutil.copy(LOG_BACKUP, dist_file)

def _init_conf():
    logging.config.dictConfig(LOG_CONF)


def reset_log():
    _init_file()
    _init_conf()

_init_conf()
logger = logging.getLogger('pyfmto')
