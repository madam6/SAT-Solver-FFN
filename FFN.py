import re
from datetime import datetime


def parse_log(log_input):
    if isinstance(log_input, str):
        # If log_input is a string, treat it as the log content
        log_content = log_input
    elif os.path.isfile(log_input):
        # If log_input is a file path, read the content from the file
        with open(log_input, 'r') as file:
            log_content = file.read()
    else:
        raise ValueError("Invalid input. Please provide a valid file path or log content as a string.")

    # The rest of the function remains unchanged
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    id_pattern = re.compile(r'Solver Result: \(([^\)]+)')
    flips_var_pattern = re.compile(r'flips_var\': (\d+\.\d+)')
    decay_factor_var_pattern = re.compile(r'decay_factor_var\': (\d+\.\d+)')
    flips_pattern = re.compile(r'flips\': (\d+)')
    success_time_pattern = re.compile(r'succeeded in (\d+\.\d+) seconds\.')
    good_result_pattern = re.compile(r'Good result: (\w+)')

    timestamp_match = re.search(timestamp_pattern, log_content)
    if timestamp_match:
        solver_id = timestamp_match.group(1)
    else:
        solver_id = 'default_id'

    flips_var = float(re.search(flips_var_pattern, log_content).group(1))
    decay_factor_var = float(re.search(decay_factor_var_pattern, log_content).group(1))
    flips = int(re.search(flips_pattern, log_content).group(1))

    success_time_match = re.search(success_time_pattern, log_content)
    running_time = float(success_time_match.group(1)) if success_time_match else None

    good_result_str = re.search(good_result_pattern, log_content)

    good_result = True if good_result_str and 'Good' in good_result_str.group() else False
    time_str = re.search(timestamp_pattern, log_content).group(1)

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


def process_log_file(log_path, good_result_weight=0.66, running_time_weight=0.165, flips_weight=0.165):
    with open(log_path, 'r') as file:
        log_content = file.read()

    # Split the log content into individual entries
    log_entries = log_content.split('\n')

    # Initialize an empty list to store the results
    solver_results_list = []

    # Process each entry in the log file
    for i in range(0, len(log_entries)-1, 5):
        # Call parse_log for each entry
        entry_string = '\n'.join(log_entries[i:i + 5])

        # Call parse_log for each entry
        result_dict = parse_log(entry_string)

        # Call add_success_score for each entry
        add_success_score(result_dict, good_result_weight, running_time_weight, flips_weight)

        # Append the result_dict to the list
        solver_results_list.append(result_dict)
    return solver_results_list

log_path = 'A:\projects for python\SAT Solver + FFN\solver_test_log.txt'
final_result = process_log_file(log_path)
for re in final_result:
    print("{\n")
    for r in re:
        print(r + " " + str(re[r]))
    print("\n}")

