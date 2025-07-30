import unittest

from tests.framework import OfflineServer


class TestEmptyServer(unittest.TestCase):

    def test_attributes(self):
        server = OfflineServer()
        self.assertEqual(server.num_clients, 0)
        self.assertTrue(list(server.sorted_ids) == [])
        self.assertEqual(OfflineServer.__doc__, None)