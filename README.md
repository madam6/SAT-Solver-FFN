# SAT Solver with Neural Network Scoring

This project implements a SAT solver (walkSAT) and utilises neural network for learning best parameters of the solver. The solver aims to determine the satisfiability of SAT (Boolean satisfiability problem) instances and provide assignments if satisfiable.

## Project Structure

The project is structured as follows:

- `Solver.py`: Contains the implementation of the SAT solver with scores from neural-net.
- `RunSolver.py`: Code for running solver on sat instances that are present in current directory.
- `RunAndLog.py`: Code for running solver and logging results into `solver_test_log.txt` utilises [PySAT](https://pysathq.github.io/) library to validate the correctness.
- `ClauseGenerator.py`: Tool for generating random clauses.
- `FFN.py`: Code that contains simple neural-net to learn parametres of the solver, also parses the `solver_test_log.txt`.

## Classes

### `Solver.py`

- `check_sat(clauses)`: Function to check the satisfiability of a set of clauses using the neural network scoring system.
#### Example usage:
```python
clauses = [[1, 2, -3], [-1, 3], [-2, 3]]
result = check_sat(clauses)
```

### RunSolver.py

#### Description:

The `RunSolver.py` script is designed to generate and solve SAT instances using the SAT solver implemented in the `Solver.py` module. It provides functions to read clauses from SAT instance files, generate test instances, and run the solver on these instances.

#### Functions:

##### `read_clauses_from_file(file_path)`

- Description:
  This function reads clauses from a SAT instance file. It skips the header line and parses each line to extract the clauses.

- Parameters:
  - `file_path` (str): Path to the SAT instance file.

- Returns:
  - `clauses` (list): List of clauses read from the file.

##### `generate_tests_and_run_solver(num_tests)`

- Description:
  This function generates and solves SAT instances. It iterates over the specified number of test instances, reads clauses from corresponding files, calls the SAT solver with the read clauses, and prints the result of each test.

- Parameters:
  - `num_tests` (int): Number of SAT instances to generate and solve.

- Example Usage:

```python
num_tests_to_generate = 1
generate_tests_and_run_solver(num_tests_to_generate)
```
### `RunAndLog.py`

#### `read_clauses_from_file(file_path)`

- Description:
  This function reads clauses from a SAT instance file.

- Parameters:
  - `file_path` (str): Path to the SAT instance file.

- Returns:
  - `clauses` (list): List of clauses read from the file.

#### `solve_with_pysat(clauses)`

- Description:
  This function solves the SAT problem using the PySAT library.

- Parameters:
  - `clauses` (list): List of clauses.

- Returns:
  - `satisfiable` (bool): Boolean indicating satisfiability.
  - `model` (list): Model satisfying the clauses if `satisfiable` is True, otherwise an empty list.

#### `TestSATSolver(unittest.TestCase)`

- Description:
  Test class for the SAT solver.

  ##### Methods:
  - `run_solver_and_assert(file_path)`
    - Description: Run the solver and assert the result.
    - Parameters:
      - `file_path` (str): Path to the SAT instance file.
    - Asserts:
      - The result returned by the solver matches the result obtained using PySAT.
      - Logs the result and time taken by the solver.

  - `test_solver_on_generated_instances()`
    - Description: Test the solver on generated SAT instances.
    - Asserts: Runs the solver on a set of generated instances and compares the results with PySAT solver.
### `ClauseGenerator.py`

#### `generate_clause(num_variables, clause_length)`

- Description:
  This function generates a random SAT clause with the specified number of variables and clause length.

- Parameters:
  - `num_variables` (int): The number of variables in the SAT instance.
  - `clause_length` (int): The length of the generated clause.

- Returns:
  - `clause` (list): A list representing the generated SAT clause.

#### `generate_clauses(num_variables, num_clauses, clause_length)`

- Description:
  This function generates a list of random SAT clauses.

- Parameters:
  - `num_variables` (int): The number of variables in the SAT instance.
  - `num_clauses` (int): The number of clauses to generate.
  - `clause_length` (int): The length of each generated clause.

- Returns:
  - `clauses` (list): A list of lists representing the generated SAT clauses.

#### `generate_files(num_files, max_variables, num_clauses, clause_length)`

- Description:
  This function generates multiple SAT instances with varying numbers of variables and the same number of clauses.

- Parameters:
  - `num_files` (int): The number of files to generate.
  - `max_variables` (int): The maximum number of variables in any generated instance.
  - `num_clauses` (int): The number of clauses in each generated instance.
  - `clause_length` (int): The length of each generated clause.

#### Example Usage:

```python
num_files_to_generate = 400
max_variables_per_instance = 100
num_clauses_per_instance = 500
clause_length_per_instance = 200

generate_files(num_files_to_generate, max_variables_per_instance, num_clauses_per_instance, clause_length_per_instance)
```

### `FFN.py`

#### `parse_log(log_input)`

- Description:
  This function parses log content to extract relevant information about solver results.

- Parameters:
  - `log_input` (str or file path): Content of the log or path to the log file.

- Returns:
  - `result_data` (dict): Dictionary containing parsed information about solver results.

#### `add_success_score(result_dict, good_result_weight=0.66, running_time_weight=0.165, flips_weight=0.165)`

- Description:
  This function calculates the success score based on specified weights and updates the result dictionary.

- Parameters:
  - `result_dict` (dict): Dictionary containing solver result information.
  - `good_result_weight` (float): Weight assigned to good results.
  - `running_time_weight` (float): Weight assigned to running time.
  - `flips_weight` (float): Weight assigned to flips.

- Returns:
  - `result_dict` (dict): Updated result dictionary with success score.

#### `process_log_file(log_path, good_result_weight=0.66, running_time_weight=0.165, flips_weight=0.165)`

- Description:
  This function processes a log file to extract and calculate information about solver results.

- Parameters:
  - `log_path` (str): Path to the log file.
  - `good_result_weight` (float): Weight assigned to good results.
  - `running_time_weight` (float): Weight assigned to running time.
  - `flips_weight` (float): Weight assigned to flips.

- Returns:
  - `solver_results_list` (list): List of dictionaries containing processed information about solver results.
## Rest of the code is an implementation of neural network.
  - To teach it, just run the whole file, note that you need to specify the number of files on which you train the model here:
  - ```python
    labels_tensor = torch.FloatTensor(labels).view(880, 1)  # Reshape to (num_samples, 1)
    ```

## Usage

1. **Install Dependencies**:
   Ensure that Python and the required packages are installed. You can install the dependencies using:
  ```bash
  pip install torch numpy scikit-learn python-sat
  ```


2. **Training the Neural Network** (Optional):
If you want to train the neural network model, run the `FFN.py` file and put the necessary variable values inside the `Solver.py`.


3. **Running the Solver**:
To use the SAT solver with the neural network scoring system, simply import the `check_sat` function from `Solver.py` and provide it with a set of clauses to check satisfiability as described above.


## License

This project is licensed under the MIT License.

## Acknowledgments

This project utilizes the [PySAT](https://pysathq.github.io/) library for verifying the resultls. Special thanks to the developers of PySAT for their valuable contribution.


