import unittest
from pyfmto.framework.packages import (
    Actions,
    ClientPackage,
    ServerPackage,
    DataArchive
)


class TestPackagesForCoverage(unittest.TestCase):
    def test_actions_enum(self):
        self.assertEqual(Actions.REGISTER.name, 'REGISTER')
        self.assertEqual(Actions.QUIT.name, 'QUIT')

    def test_client_package_init(self):
        pkg = ClientPackage(cid=1, action=Actions.REGISTER, data={"key": "value"})
        self.assertEqual(pkg.cid, 1)
        self.assertEqual(pkg.action, Actions.REGISTER)
        self.assertEqual(pkg.data, {"key": "value"})

    def test_client_package_no_data(self):
        pkg = ClientPackage(cid=None, action=Actions.QUIT)
        self.assertIsNone(pkg.cid)
        self.assertEqual(pkg.action, Actions.QUIT)
        self.assertIsNone(pkg.data)

    def test_server_package(self):
        pkg = ServerPackage(desc="Test description", data=[1, 2, 3])
        self.assertEqual(pkg.desc, "Test description")
        self.assertEqual(pkg.data, [1, 2, 3])

    def test_server_package_no_data(self):
        pkg = ServerPackage(desc="Empty data")
        self.assertEqual(pkg.desc, "Empty data")
        self.assertIsNone(pkg.data)

    def test_data_archive_initial_state(self):
        archive = DataArchive()
        self.assertEqual(archive.num_src, 0)
        self.assertEqual(archive.num_res, 0)
        self.assertIsNone(archive.get_latest_src())
        self.assertIsNone(archive.get_latest_res())

    def test_data_archive_add_and_get(self):
        archive = DataArchive()

        # Add src data
        archive.add_src("src1")
        archive.add_src("src2")
        self.assertEqual(archive.num_src, 2)
        self.assertEqual(archive.get_latest_src(), "src2")

        # Add res data
        archive.add_res("res1")
        archive.add_res("res2")
        self.assertEqual(archive.num_res, 2)
        self.assertEqual(archive.get_latest_res(), "res2")
