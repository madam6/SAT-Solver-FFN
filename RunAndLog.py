import unittest
import time
import logging
from Solver import check_sat
from pysat.solvers import Glucose3

logging.basicConfig(filename='solver_test_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def read_clauses_from_file(file_path):
    """
       Read clauses from a SAT instance file.

       Args:
       file_path (str): Path to the SAT instance file.

       Returns:
       list: List of clauses.
    """
    clauses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header line
            clause = list(map(int, line.strip().split()[:-1]))
            clauses.append(clause)
    return clauses


def solve_with_pysat(clauses):
    """
       Solve SAT problem using PySAT.

       Args:
       clauses (list): List of clauses.

       Returns:
       tuple: Tuple containing a boolean indicating satisfiability and the model.
    """
    with Glucose3() as solver:
        for clause in clauses:
            solver.add_clause(clause)
        return solver.solve(), solver.get_model()


class TestSATSolver(unittest.TestCase):
    """
        Test class for SAT solver.
    """
    def run_solver_and_assert(self, file_path):
        """
               Run the solver and assert the result.

               Args:
               file_path (str): Path to the SAT instance file.
        """
        generated_clauses = read_clauses_from_file(file_path)

        # Measure the time taken by the solver
        start_time = time.time()
        result_solver = check_sat(generated_clauses)
        elapsed_time = time.time() - start_time

        result_pysat_solver, assignment_pysat = solve_with_pysat(generated_clauses)

        # Log the result and time
        if result_solver is not None:
            logging.info(f"Assignment for {file_path} succeeded in {elapsed_time:.4f} seconds.")
        else:
            logging.error(f"Assignment for {file_path} failed.")

        # Add a separate log for the result of the assignment
        logging.info(f"My Solver Result: {result_solver}")
        logging.info(f"PySAT Solver Result: {result_pysat_solver}")
        logging.info(f"PySAT Solver Assignment: {assignment_pysat}")

        # Check if both assignments are None or if they are equal
        if result_solver[0] is None and result_pysat_solver == False:
            logging.info("Good result: Both solvers returned None.")
        elif result_solver[0] == result_pysat_solver:
            logging.info("Good result: Both solvers returned similar assignments.")
        elif result_pysat_solver and result_solver[0] is not None:
            logging.info("Good result: Both solvers found satisfying assignment.")
        else:
            logging.info("Bad result: CNF is satisfiable, my solver could not find a solution.")

        self.assertEqual(result_solver is not None, result_pysat_solver)

    def test_solver_on_generated_instances(self):
        num_tests_to_generate = 400

        for test_index in range(1, num_tests_to_generate + 1):
            file_path = f'sat_instance_{test_index}.cnf'
            with self.subTest(file_path=file_path):
                self.run_solver_and_assert(file_path)


if __name__ == '__main__':
    # Run the tests
    unittest.main()
