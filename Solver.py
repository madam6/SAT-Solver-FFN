import random
from typing import List, Set, Dict

# Part A.1
# Worst case complexity: O(n)
# Best case complexity: O(1)
def check_clause(assignment, clause):
    check_clause_result = False
    for i in range(len(clause)):
        pos_in_assignment = abs(clause[i])
        if (clause[i] > 0 and assignment[pos_in_assignment] == 1) or \
           (clause[i] < 0 and assignment[pos_in_assignment] == -1):
            check_clause_result = True
            break
    return check_clause_result

# Part A.2
# Worst case complexity: O(N * L) where N - amount of clauses in database, L - maximum length of clause
# Best case complexity: O(1) when clauseDatabase is empty - it returns False
def check_clause_database(assignment, clause_database):
    check_clause_db_result = False
    checks = [check_clause(assignment, clause) for clause in clause_database]
    for check in checks:
        if not check:
            check_clause_db_result = False
            break
        check_clause_db_result = True
    return check_clause_db_result

# Part A.3
# Worst case complexity: O(n)
# Best case complexity: O(1)
def check_clause_partial(partial_assignment, clause):
    check_clause_result = False
    is_there_a_0 = False
    count_zero = len(partial_assignment) - 1

    for i in range(len(clause)):
        pos_in_assignment = abs(clause[i])
        if (clause[i] > 0 and partial_assignment[pos_in_assignment] == 1) or \
           (clause[i] < 0 and partial_assignment[pos_in_assignment] == -1):
            check_clause_result = True
            break
        elif partial_assignment[pos_in_assignment] == 0:
            count_zero -= 1
        if count_zero == 0:
            is_there_a_0 = True

    return 1 if check_clause_result else (0 if is_there_a_0 else -1)

# Part A.4
# Worst case complexity: O(n)
# Best case complexity: O(n)
def find_unit(partial_assignment, clause):
    is_break = False
    count_zero = 0
    temp_index = -1

    for i in range(1, len(partial_assignment)):
        if partial_assignment[i] == 0:
            count_zero += 1
            temp_index = i
        if partial_assignment[i] == -1:
            if clause[i - 1] > 0:
                continue
            else:
                is_break = True
                break
        if partial_assignment[i] == 1:
            if clause[i - 1] < 0:
                continue
            else:
                is_break = True
                break

    if is_break:
        return 0
    else:
        return clause[temp_index - 1] if count_zero == 1 else 0

# Part B
def check_sat(clause_database):
    flips = get_amount_of_flips(len(clause_database), get_number_of_variables(clause_database))
    clauses = [set(clause) for clause in clause_database]

    num_variables = get_number_of_variables(clause_database)
    assignment = [-1] * (num_variables + 1)
    variable_occurrences = count_vars(clause_database)
    most_frequent_variable = get_most_frequent_variable(variable_occurrences, assignment)
    least_frequent_variable = get_least_frequent_variable(variable_occurrences)
    assignment[0] = 0
    assignment[abs(most_frequent_variable)] = 1
    assignment[abs(least_frequent_variable)] = -1

    random.seed()
    for _ in range(flips):
        if check_clause_database(clauses, assignment):
            return assignment
        unsatisfied_clause_index = get_random_unsatisfied_clause_index(clauses, assignment)
        if random.random() < 0.7:
            flip_variable = get_next_variable(clauses, assignment, 0.7)
            assignment[abs(flip_variable)] *= -1
        else:
            flip_variable = get_most_frequent_variable(clauses[unsatisfied_clause_index], assignment)
            assignment[abs(flip_variable)] *= -1

        for j in range(1, num_variables + 1):
            score = 0
            for clause in clauses:
                if j in clause and not check_clause(clause, assignment):
                    score += 1
                elif -j in clause and not check_clause(clause, assignment):
                    score -= 1
            if score > 0:
                assignment[j] = 1
            elif score < 0:
                assignment[j] = -1

    if check_clause_database(clauses, assignment):
        return assignment
    else:
        return None

def get_amount_of_flips(num_clauses, num_vars):
    num_flips = 0
    clause_ratio = num_clauses / num_vars

    if clause_ratio < 2.0:
        num_flips = int((num_vars / 2) * 8 ** clause_ratio)
    elif clause_ratio < 4.0:
        num_flips = int((num_vars / 3) * 8 ** clause_ratio)
    elif clause_ratio < 6.0:
        num_flips = 8500
    elif clause_ratio < 8.0:
        num_flips = 8500
    else:
        num_flips = int((num_vars / 6) * 8 ** clause_ratio)

    return num_flips

def get_most_frequent_variable(variable_occurrences):
    most_frequent = 0
    most_frequent_count = 0
    for variable, count in variable_occurrences.items():
        if count > most_frequent_count:
            most_frequent = variable
            most_frequent_count = count
    return most_frequent

def count_vars(clause_database):
    variable_occurrences = {}
    for clause in clause_database:
        for variable in clause:
            variable_occurrences[variable] = variable_occurrences.get(variable, 0) + 1
    return variable_occurrences

def check_clause_database(clause_database, assignment):
    for clause in clause_database:
        clause_satisfied = False
        for variable in clause:
            if (variable > 0 and assignment[variable] == 1) or \
               (variable < 0 and assignment[-variable] == -1):
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False
    return True

def get_random_unsatisfied_clause_index(clause_database, assignment):
    unsatisfied_clause_indices = [
        i for i, clause in enumerate(clause_database) if not check_clause(clause, assignment)
    ]
    return random.choice(unsatisfied_clause_indices)

def check_clause(clause, assignment):
    for variable in clause:
        if (variable > 0 and assignment[variable] == 1) or \
           (variable < 0 and assignment[-variable] == -1):
            return True
    return False

def get_most_frequent_variable(clause, assignment):
    variable_frequency = {}
    for variable in clause:
        if (variable > 0 and assignment[variable] == 1) or \
           (variable < 0 and assignment[-variable] == -1):
            continue
        variable_frequency[variable] = variable_frequency.get(variable, 0) + 1
    most_frequent = 0
    most_frequent_count = 0
    for variable, count in variable_frequency.items():
        if count > most_frequent_count:
            most_frequent = variable
            most_frequent_count = count
    return most_frequent

def get_least_frequent_variable(variable_occurrences):
    least_frequent = 0
    least_frequent_count = float('inf')
    for variable, count in variable_occurrences.items():
        if count < least_frequent_count:
            least_frequent = variable
            least_frequent_count = count
    return least_frequent

def get_next_variable(clause_database, assignment, decay_factor):
    variable_scores = {}
    for clause in clause_database:
        if check_clause(clause, assignment):
            continue
        for variable in clause:
            if (variable > 0 and assignment[variable] == 1) or \
               (variable < 0 and assignment[-variable] == -1):
                continue
            variable_scores[variable] = variable_scores.get(variable, 0.0) + 1.0
    for i in range(1, len(assignment)):
        variable_scores[i] = variable_scores.get(i, 0.0) * decay_factor
        variable_scores[-i] = variable_scores.get(-i, 0.0) * decay_factor
    next_variable = 0
    max_score = -1
    for variable, score in variable_scores.items():
        if score > max_score:
            next_variable = variable
            max_score = score
    return next_variable

def get_number_of_variables(cnf):
    max_var = 0
    for clause in cnf:
        for lit in clause:
            var = abs(lit)
            if var > max_var:
                max_var = var
    return max_var
