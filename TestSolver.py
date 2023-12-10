import unittest
from Solver import check_sat
from pysat.solvers import Glucose3


def read_clauses_from_file(file_path):
    clauses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header line
            clause = list(map(int, line.strip().split()[:-1]))
            clauses.append(clause)
    return clauses


def solve_with_pysat(clauses):
    with Glucose3() as solver:
        for clause in clauses:
            solver.add_clause(clause)
        return solver.solve()


class TestSATSolver(unittest.TestCase):

    def run_solver_and_assert(self, file_path):
        generated_clauses = read_clauses_from_file(file_path)

        result_your_solver = check_sat(generated_clauses)
        result_pysat_solver = solve_with_pysat(generated_clauses)

        self.assertEqual(result_your_solver is not None, result_pysat_solver)

    def test_solver_on_generated_instances(self):
        num_tests_to_generate = 2

        for test_index in range(1, num_tests_to_generate + 1):
            file_path = f'sat_instance_{test_index}.cnf'
            with self.subTest(file_path=file_path):
                self.run_solver_and_assert(file_path)


if __name__ == '__main__':
    unittest.main()
