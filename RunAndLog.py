import os
import logging

from RunSolver import read_clauses_from_file
from Solver import check_sat

# Set up logging
logging.basicConfig(filename='solver_log.txt', level=logging.INFO)

# Directory where SAT instance files are located
instance_directory = '.'  # Change this if your instance files are in a different directory
num_tests = 2  # Change this to the number of test instances

# Loop through each test instance
for test_index in range(1, num_tests + 1):
    file_path = f'sat_instance_{test_index}.cnf'

    # Read clauses from file


    generated_clauses = read_clauses_from_file(file_path)

    # Call your SAT solver with the read clauses
    result_assignment = check_sat(generated_clauses)

    logging.info(f"Instance: {file_path}")

    if result_assignment is not None:
        logging.info("Solver ran successfully. Variables used:")
        for variable, value in enumerate(result_assignment):
            if variable != 0:  # Ignore the 0 index
                logging.info(f"Variable {variable}: {value}")
    else:
        logging.info("Solver failed to find a satisfying assignment.")
