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


def generate_tests_and_run_solver(num_tests):
    for test_index in range(1, num_tests + 1):
        file_path = f'sat_instance_{test_index}.cnf'

        # Read clauses from file
        generated_clauses = read_clauses_from_file(file_path)

        # Call your SAT solver with the read clauses
        result_your_solver = check_sat(generated_clauses)

        # Call PySAT solver with the read clauses
        result_pysat_solver = solve_with_pysat(generated_clauses)

        # Print the results
        print(
            f"Test {test_index}: Your Solver - {'SAT' if result_your_solver is not None else 'UNSAT'}, PySAT Solver - {'SAT' if result_pysat_solver else 'UNSAT'}")


# Example usage:
num_tests_to_generate = 5

generate_tests_and_run_solver(num_tests_to_generate)
