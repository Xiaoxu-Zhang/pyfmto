# Reference:
#   https://www.bilibili.com/video/BV1sK4y1x7e1
#   https://www.cnblogs.com/kangshuaibo/p/14700833.html

import logging.config
import time
from pathlib import Path
from concurrent_log_handler import ConcurrentRotatingFileHandler

log_dir = Path("out/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "pyfmto.log"


class SafeFileHandler(ConcurrentRotatingFileHandler):
    def rotation_filename(self, default_name: str) -> str:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = Path(self.baseFilename)
        return str(filename.with_name(f"{filename.stem}_{ts}.log"))


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
            '()': SafeFileHandler,
            'level': 'DEBUG',
            'formatter': 'simpleFormatter',
            'filename': 'out/logs/pyfmto.log',
            'maxBytes': 2 * 1024 * 1024,
            'backupCount': 10,
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        'pyfmto': {
            'level': 'DEBUG',
            'handlers': ['pyfmto_handler'],
            'propagate': False
        }
    }
}

logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger('pyfmto')
