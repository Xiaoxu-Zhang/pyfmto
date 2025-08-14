import numpy as np
import unittest

from pyfmto.problems.solution import Solution


class TestSolution(unittest.TestCase):
    def setUp(self):
        """Setup common test environment."""
        self.dim = 20
        self.obj = 1
        self.init_fe = 5 * self.dim
        self.max_fe = 11 * self.dim
        self.np_per_dim = 1
        self.solution_empty = Solution()
        self.solution_initialized = Solution()
        x = np.random.random(size=(self.init_fe, self.dim))
        y = np.random.random(size=(self.init_fe, self.obj))
        self.solution_initialized.append(x, y)

    def test_empty_solutions_attributes(self):
        """Test that solution initialization sets correct attributes."""
        self.assertEqual(self.solution_empty.dim, -1)
        self.assertEqual(self.solution_empty.obj, -1)
        self.assertEqual(self.solution_empty.fe_init, -1)
        self.assertEqual(self.solution_empty.x.size, 0)
        self.assertEqual(self.solution_empty.y.size, 0)
        self.assertEqual(self.solution_empty.x_global.size, 0)
        self.assertEqual(self.solution_empty.y_global.size, 0)
        self.assertFalse(self.solution_empty.initialized)
        self.assertEqual(self.solution_empty.size, 0)

    def test_append_invalid_shape_raises_value_error(self):
        """Test appending datasets with mismatched shapes raises ValueError."""
        x, y = [1, 2, 3], [1]
        self.assertRaises(ValueError, self.solution_empty.append, x, y)
        x = np.random.random(size=(10, self.dim))
        y = np.random.random(size=(20, self.obj))
        self.assertRaises(ValueError, self.solution_empty.append, x, y)

    def test_append_not_match_dim_raise_value_error(self):
        """Test appending to empty with datasets that size mismatching to init_fe raises ValueError."""
        x = np.random.random(size=(1, self.dim + 1))
        y = np.random.random(size=(1, self.obj))
        self.assertRaises(ValueError, self.solution_initialized.append, x, y)
        x = np.random.random(size=(1, self.dim))
        y = np.random.random(size=(1, self.obj + 1))
        self.assertRaises(ValueError, self.solution_initialized.append, x, y)

    def test_information_attributes(self):
        _ = self.solution_empty.__str__()
        _ = self.solution_empty.__repr__()

    def test_initialized_solution_sets_correct_attributes(self):
        """Test that a solution can be initialized with correct attributes."""
        self.assertTrue(self.solution_initialized.initialized)
        self.assertEqual(self.solution_initialized.dim, self.dim)
        self.assertEqual(self.solution_initialized.obj, self.obj)
        self.assertEqual(self.solution_initialized.fe_init, self.init_fe)
        self.assertEqual(self.solution_initialized.size, self.init_fe)
        self.assertEqual(self.solution_initialized.num_updated, self.init_fe)

    def test_init_from_dict_recreates_same_object(self):
        """Test that a solution can be recreated from its own dictionary."""
        sol_dict = self.solution_initialized.to_dict()
        solution = Solution(solution=sol_dict)

        self.assertEqual(solution.dim, self.solution_initialized.dim)
        self.assertEqual(solution.obj, self.solution_initialized.obj)
        self.assertEqual(solution.fe_init, self.solution_initialized.fe_init)
        self.assertTrue(np.all(solution.x == self.solution_initialized.x))
        self.assertTrue(np.all(solution.y == self.solution_initialized.y))

    def test_multiple_appends_accumulate_correctly(self):
        """Test multiple appends accumulate the datasets correctly."""
        self.assertEqual(self.solution_initialized.num_updated, self.init_fe)
        x1 = np.random.random(size=(10, self.dim))
        y1 = np.random.random(size=(10, self.obj))
        self.solution_initialized.append(x1, y1)
        self.assertEqual(self.solution_initialized.size, self.init_fe + x1.shape[0])
        self.assertEqual(self.solution_initialized.num_updated, x1.shape[0])

        x2 = np.random.random(size=(5, self.dim))
        y2 = np.random.random(size=(5, self.obj))
        self.solution_initialized.append(x2, y2)
        self.assertEqual(self.solution_initialized.num_updated, x2.shape[0])
        self.assertEqual(self.solution_initialized.size, self.init_fe + x1.shape[0] + x2.shape[0])
        self.assertEqual(self.solution_initialized.x.shape[0], self.solution_initialized.size)
        self.assertEqual(self.solution_initialized.y.shape[0], self.solution_initialized.size)

    def test_properties_return_expected_values_for_single_objective(self):
        """Test property values for single objective case."""
        self.assertEqual(self.solution_initialized.y_max, np.max(self.solution_initialized.y))
        self.assertEqual(self.solution_initialized.y_min, np.min(self.solution_initialized.y))

        idx_max = np.argmax(self.solution_initialized.y.flatten())
        self.assertTrue(np.array_equal(self.solution_initialized.x_of_y_max, self.solution_initialized.x[idx_max]))

        idx_min = np.argmin(self.solution_initialized.y.flatten())
        self.assertTrue(np.array_equal(self.solution_initialized.x_of_y_min, self.solution_initialized.x[idx_min]))

    def test_homo_sequences_start_end_with_expected_values(self):
        """Test homogenized sequences start with initial y value and end with max or min value."""
        self.assertEqual(self.solution_initialized.y_homo_decrease[0], self.solution_initialized.y[0])
        self.assertEqual(self.solution_initialized.y_homo_increase[0], self.solution_initialized.y[0])
        self.assertEqual(self.solution_initialized.y_homo_decrease[-1], self.solution_initialized.y_min)
        self.assertEqual(self.solution_initialized.y_homo_increase[-1], self.solution_initialized.y_max)

    def test_property_access_raises_error_in_multi_objective_mode(self):
        """Test accessing single-objective properties in multi-objective mode raises error."""
        solution = Solution()
        x = np.random.random(size=(self.init_fe, self.dim))
        y = np.random.random(size=(self.init_fe, 2))
        solution.append(x, y)

        with self.assertRaises(AttributeError):
            _ = solution.y_max
        with self.assertRaises(AttributeError):
            _ = solution.y_min
        with self.assertRaises(AttributeError):
            _ = solution.x_of_y_max
        with self.assertRaises(AttributeError):
            _ = solution.x_of_y_min
        with self.assertRaises(AttributeError):
            _ = solution.y_homo_decrease
        with self.assertRaises(AttributeError):
            _ = solution.y_homo_increase
