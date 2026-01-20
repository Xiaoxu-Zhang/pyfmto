import logging.config
from unittest.mock import patch

from pyfmto.utilities.loggers import LOG_CONF, SafeFileHandler
from tests.helpers import PyfmtoTestCase


class TestPyfmtoRotatingFileHandler(PyfmtoTestCase):
    def setUp(self):
        super().setUp()
        # Configure logging
        logging.config.dictConfig(LOG_CONF)
        self.logger = logging.getLogger('pyfmto')

    @patch("time.strftime", return_value="2023-10-10_12-00-00")
    def test_rotation_filename(self, mock_strftime):
        handler = self.logger.handlers[0]
        self.assertIsInstance(handler, SafeFileHandler)
        default_name = "pyfmto.log"
        rotated_name = handler.rotation_filename(default_name)
        expected_name = str(self.log_dir / "pyfmto_2023-10-10_12-00-00.log")
        self.assertEqual(rotated_name[-len(expected_name):], expected_name)
