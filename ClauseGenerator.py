import random

def generate_clause(num_variables, clause_length):
    """
    Generate a random SAT clause.

    Parameters:
    - num_variables (int): The number of variables in the SAT instance.
    - clause_length (int): The length of the generated clause.

    Returns:
    - list: A list representing the generated SAT clause.
    """
    clause_length = min(clause_length, num_variables)
    clause = random.sample(range(1, num_variables + 1), clause_length)
    clause = [var if random.choice([True, False]) else -var for var in clause]
    return clause

def generate_clauses(num_variables, num_clauses, clause_length):
    """
    Generate a list of random SAT clauses.

    Parameters:
    - num_variables (int): The number of variables in the SAT instance.
    - num_clauses (int): The number of clauses to generate.
    - clause_length (int): The length of each generated clause.

    Returns:
    - list: A list of lists representing the generated SAT clauses.
    """
    return [generate_clause(num_variables, clause_length) for _ in range(num_clauses)]

def generate_files(num_files, max_variables, num_clauses, clause_length):
    """
    Generate multiple SAT instances with varying numbers of variables and the same number of clauses.

    Parameters:
    - num_files (int): The number of files to generate.
    - max_variables (int): The maximum number of variables in any generated instance.
    - num_clauses (int): The number of clauses in each generated instance.
    - clause_length (int): The length of each generated clause.

    Returns:
    - None
    """
    for file_index in range(1, num_files + 1):
        num_variables = random.randint(1, max_variables)
        clauses = generate_clauses(num_variables, num_clauses, clause_length)

        with open(f'sat_instance_{file_index}.cnf', 'w') as file:
            file.write(f'p cnf {num_variables} {num_clauses}\n')
            for clause in clauses:
                file.write(' '.join(map(str, clause)) + ' 0\n')


num_files_to_generate = 400
max_variables_per_instance = 100
num_clauses_per_instance = 500
clause_length_per_instance = 200

generate_files(num_files_to_generate, max_variables_per_instance, num_clauses_per_instance, clause_length_per_instance)