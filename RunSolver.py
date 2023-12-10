from Solver import check_sat


def read_clauses_from_file(file_path):
    clauses = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # Skip the header line
            clause = list(map(int, line.strip().split()[:-1]))
            clauses.append(clause)
    return clauses


def generate_tests_and_run_solver(num_tests):
    for test_index in range(1, num_tests + 1):
        file_path = f'sat_instance_{test_index}.cnf'

        # Read clauses from file
        generated_clauses = read_clauses_from_file(file_path)

        # Call your SAT solver with the read clauses
        result = check_sat(generated_clauses)

        # Print the result
        print(f"Test {test_index}: {'SAT' if result is not None else 'UNSAT'}")
        if result is not None:
            print(f"Assignment: {result}")


# Example usage:
num_tests_to_generate = 2

generate_tests_and_run_solver(num_tests_to_generate)
