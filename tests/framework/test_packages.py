import shutil
import unittest

from pyfmto.framework.packages import (
    Actions,
    ClientPackage,
    DataArchive,
    SyncDataManager
)


class TestPackagesForCoverage(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree('out', ignore_errors=True)

    def test_actions_enum(self):
        self.assertEqual(Actions.REGISTER.name, 'REGISTER')
        self.assertEqual(Actions.QUIT.name, 'QUIT')

    def test_client_package_init(self):
        pkg = ClientPackage(cid=1, action=Actions.REGISTER)
        self.assertEqual(pkg.cid, 1)
        self.assertEqual(pkg.action, Actions.REGISTER)

    def test_client_package_no_data(self):
        pkg = ClientPackage(cid=None, action=Actions.QUIT)
        self.assertIsNone(pkg.cid)
        self.assertEqual(pkg.action, Actions.QUIT)

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


class TestSyncDataManager(unittest.TestCase):

    def setUp(self):
        self.sync_manager = SyncDataManager()

    def tearDown(self):
        shutil.rmtree('out', ignore_errors=True)

    def test_update_src(self):
        self.sync_manager.update_src(1, 100, "test_data")
        self.assertEqual(self.sync_manager.get_src(1, 100), "test_data")

    def test_update_res(self):
        self.sync_manager.update_res(1, 200, "result_data")
        self.assertEqual(self.sync_manager.get_res(1, 200), "result_data")

    def test_lts_src_ver(self):
        self.sync_manager.update_src(1, 100, "test_data")
        self.sync_manager.update_src(1, 101, "newer_data")
        self.assertEqual(self.sync_manager.lts_src_ver(1), 101)

    def test_lts_res_ver(self):
        self.sync_manager.update_res(1, 200, "result_data")
        self.sync_manager.update_res(1, 201, "newer_result")
        self.assertEqual(self.sync_manager.lts_res_ver(1), 201)

    def test_lts_src_raise(self):
        pass

    def test_get_src(self):
        self.sync_manager.update_src(1, 100, "test_data")
        self.assertEqual(self.sync_manager.get_src(1, 100), "test_data")
        self.assertIsNone(self.sync_manager.get_src(2, 100))
        self.assertIsNone(self.sync_manager.get_src(1, 101))

    def test_get_res(self):
        self.sync_manager.update_res(1, 200, "result_data")
        self.assertEqual(self.sync_manager.get_res(1, 200), "result_data")
        self.assertIsNone(self.sync_manager.get_res(2, 200))
        self.assertIsNone(self.sync_manager.get_res(1, 201))

    def test_available_src_ver(self):
        self.assertEqual(self.sync_manager.available_src_ver, -1)
        self.sync_manager.update_src(1, 100, "data1")
        self.sync_manager.update_src(1, 101, "data2")
        self.sync_manager.update_src(2, 100, "data3")
        self.sync_manager.update_src(2, 101, "data4")
        self.sync_manager.update_src(2, 102, "data5")
        self.assertEqual(self.sync_manager.available_src_ver, 101)

    def test_num_clients(self):
        self.assertEqual(self.sync_manager.num_clients, 0)
        self.sync_manager.update_src(1, 100, "data1")
        self.assertEqual(self.sync_manager.num_clients, 1)
        self.sync_manager.update_src(2, 200, "data2")
        self.assertEqual(self.sync_manager.num_clients, 2)
