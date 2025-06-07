import unittest
import logging
import shutil

from pyfmto.utilities import loggers


class TestLoggers(unittest.TestCase):

    def tearDown(self):
        if loggers.LOG_PATH.exists():
            shutil.rmtree(loggers.LOG_PATH)

    def test_check_files_creates_log_files_with_header(self):
        loggers._init_file()
        self.assertTrue(loggers.LOG_FILE.exists())
        with open(loggers.LOG_FILE, 'r', encoding='utf-8') as f:
            content = f.read(len(loggers.LOG_HEAD))
            self.assertEqual(content, loggers.LOG_HEAD)

    def test_init_conf_loads_config_correctly(self):
        logger = logging.getLogger('client_logger')
        self.assertIsNotNone(logger)
        self.assertTrue(logger.hasHandlers())

    def test_reset_log_files_backs_up_and_resets_all(self):
        loggers.reset_log()
        with open(loggers.LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("Original Content")

        loggers.reset_log()

        self.assertTrue(loggers.LOG_BACKUP.exists())
        with open(loggers.LOG_BACKUP, 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "Original Content")
