import unittest
import time
import logging
from Solver import check_sat
from pysat.solvers import Glucose3

logging.basicConfig(filename='solver_test_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')
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
        return solver.solve(), solver.get_model()


class TestSATSolver(unittest.TestCase):

    def run_solver_and_assert(self, file_path):
        generated_clauses = read_clauses_from_file(file_path)

        # Measure the time taken by the solver
        start_time = time.time()
        result_your_solver = check_sat(generated_clauses)
        elapsed_time = time.time() - start_time

        result_pysat_solver, assignment_pysat = solve_with_pysat(generated_clauses)

        # Log the result and time
        if result_your_solver is not None:
            logging.info(f"Assignment for {file_path} succeeded in {elapsed_time:.4f} seconds.")
        else:
            logging.error(f"Assignment for {file_path} failed.")

        # Add a separate log for the result of the assignment
        logging.info(f"Your Solver Result: {result_your_solver}")
        logging.info(f"PySAT Solver Result: {result_pysat_solver}")
        logging.info(f"PySAT Solver Assignment: {assignment_pysat}")

        self.assertEqual(result_your_solver is not None, result_pysat_solver)

    def test_solver_on_generated_instances(self):
        num_tests_to_generate = 5

        for test_index in range(1, num_tests_to_generate + 1):
            file_path = f'sat_instance_{test_index}.cnf'
            with self.subTest(file_path=file_path):
                self.run_solver_and_assert(file_path)


if __name__ == '__main__':
    # Run the tests
    unittest.main()
