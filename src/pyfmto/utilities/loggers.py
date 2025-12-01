# Reference:
#   https://www.bilibili.com/video/BV1sK4y1x7e1
#   https://www.cnblogs.com/kangshuaibo/p/14700833.html

import logging.config
import logging.handlers
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler

__all__ = ['logger']
Path('out', 'logs').mkdir(parents=True, exist_ok=True)


class PyfmtoRotatingFileHandler(RotatingFileHandler):
    def rotation_filename(self, default_name: str) -> str:
        filename = Path(self.baseFilename)
        base = filename.stem
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        new_name = filename.with_name(f"{base} {timestamp}.log")
        return str(new_name)


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
            '()': PyfmtoRotatingFileHandler,
            'level': 'DEBUG',
            'formatter': 'simpleFormatter',
            'filename': 'out/logs/pyfmto.log',
            'maxBytes': 2 * 1024 * 1024,
            'backupCount': 10
        }
    },
    'loggers': {
        'pyfmto': {
            'level': 'DEBUG',
            'handlers': ['pyfmto_handler'],
            'propagate': 0
        }
    }
}


logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger('pyfmto')
