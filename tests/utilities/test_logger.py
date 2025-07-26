import unittest
import shutil

from pyfmto.utilities import loggers, logger, reset_log


class TestLoggers(unittest.TestCase):

    def tearDown(self):
        if loggers.LOG_PATH.exists():
            shutil.rmtree(loggers.LOG_PATH)

    def test_init_conf_loads_config_correctly(self):
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

    def test_backup_to(self):
        logger.info("Testing backup")
        reset_log()
        backup_to = loggers.LOG_PATH / 'backup'
        loggers.backup_log_to(backup_to)
        self.assertTrue((backup_to / 'backup.log').exists())
