import logging.config
from unittest.mock import patch

from pyfmto.utilities.loggers import LOG_CONF, PyfmtoRotatingFileHandler
from pathlib import Path

from tests.helpers import PyfmtoTestCase


class TestPyfmtoRotatingFileHandler(PyfmtoTestCase):
    def setUp(self):
        self.log_dir = Path("out/logs")
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Configure logging
        logging.config.dictConfig(LOG_CONF)
        self.logger = logging.getLogger('pyfmto')

    def tearDown(self):
        self.delete(self.log_dir)

    @patch("time.strftime", return_value="2023-10-10 12:00:00")
    def test_rotation_filename(self, mock_strftime):
        handler = self.logger.handlers[0]
        self.assertIsInstance(handler, PyfmtoRotatingFileHandler)
        default_name = str("pyfmto.log")
        rotated_name = handler.rotation_filename(default_name)
        expected_name = str(self.log_dir / "pyfmto 2023-10-10 12:00:00.log")
        self.assertEqual(rotated_name[-len(expected_name):], expected_name)
