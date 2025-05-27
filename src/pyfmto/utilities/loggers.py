# Reference:
#   https://www.bilibili.com/video/BV1sK4y1x7e1
#   https://www.cnblogs.com/kangshuaibo/p/14700833.html

import logging.config
import logging.handlers
import shutil
from typing import Union

import yaml
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
CONF_FILE = Path(__file__).parent / 'logging.yaml'
LOG_FILE = LOG_PATH / 'pyfmto.log'
LOG_BACKUP = LOG_PATH / 'backup.log'


def _check_path():
    if not LOG_PATH.exists():
        LOG_PATH.mkdir(parents=True)


def _check_file():
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(LOG_HEAD)


def _init_conf():
    with open(CONF_FILE, 'r') as config_file:
        conf_dict = yaml.safe_load(config_file)
    conf_dict['handlers']['pyfmto_handler']['filename'] = str(LOG_PATH / 'pyfmto.log')
    logging.config.dictConfig(conf_dict)


def set_output_path(path: Union[Path, str]):
    global LOG_PATH
    LOG_PATH = Path(path)


def reset_log():
    if LOG_FILE.exists():
        shutil.copy(str(LOG_FILE), str(LOG_BACKUP))
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(LOG_HEAD)


_check_path()
_check_file()
_init_conf()

logger = logging.getLogger('pyfmto')
