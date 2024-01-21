import re
from datetime import datetime


def parse_log(log_path):
    with open(log_path, 'r') as file:
        log_content = file.read()

    # Define regular expressions to extract information
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    id_pattern = re.compile(r'Solver Result: \(([^\)]+)')
    flips_var_pattern = re.compile(r'flips_var\': (\d+\.\d+)')
    decay_factor_var_pattern = re.compile(r'decay_factor_var\': (\d+\.\d+)')
    flips_pattern = re.compile(r'flips\': (\d+)')
    success_time_pattern = re.compile(r'succeeded in (\d+\.\d+) seconds\.')

    good_result_pattern = re.compile(r'Good result: (\w+)')

    # Extract information using regular expressions
    timestamp_match = re.search(timestamp_pattern, log_content)
    if timestamp_match:
        solver_id = timestamp_match.group(1)
    else:
        solver_id = 'default_id'

    solver_id = re.search(timestamp_pattern, log_content).group(1)
    flips_var = float(re.search(flips_var_pattern, log_content).group(1))
    decay_factor_var = float(re.search(decay_factor_var_pattern, log_content).group(1))
    flips = int(re.search(flips_pattern, log_content).group(1))

    success_time_match = re.search(success_time_pattern, log_content)
    running_time = float(success_time_match.group(1)) if success_time_match else None

    good_result_str = re.search(good_result_pattern, log_content)

    # Set good_result to True if "Good result" is present, False otherwise
    good_result = True if good_result_str and 'Good' in good_result_str.group() else False
    time_str = re.search(timestamp_pattern, log_content).group(1)

    # Organize data into a dictionary
    result_data = {
        'solver_id': solver_id,
        'flips_var': flips_var,
        'decay_factor_var': decay_factor_var,
        'flips': flips,
        'good_result': good_result,
        'running_time': running_time,
    }

    return result_data

def add_success_score(result_dict, good_result_weight=0.66, running_time_weight=0.165, flips_weight=0.165):
    max_running_time = 10.0
    max_flips = 50000

    good_result = result_dict['good_result']
    running_time = result_dict['running_time']
    flips = result_dict['flips']

    success_score = (
        good_result_weight * (1 if good_result else 0) +
        running_time_weight * ((max_running_time - running_time)/max_running_time) +
        flips_weight * ((max_flips - flips)/max_flips)
    )

    result_dict['success_score'] = success_score

    return result_dict

log_path = 'A:\projects for python\SAT Solver + FFN\solver_test_log.txt'
solver_results = parse_log(log_path)
solver_results_with_succses = add_success_score(solver_results)
for result in solver_results_with_succses:
    print(result + " " + str(solver_results[result]))
