from IPython.display import display, HTML
from tabulate import tabulate
from typing import List, Set
import numpy as np
import math
import random

random.seed(42)

def pprint_table(matrix):
     display(tabulate(matrix, headers='keys', tablefmt='html'))

def pprint_list(lst: List):
     for item in lst:
        print(item, end=", ")
     print("")

def check_well_formed(df):
    # Early exit if the matrix is full of zeros
    if (df == 0).all(axis=None):  # Vectorized check for all-zero matrix
        return False
    
    # Convert the DataFrame to a NumPy array for faster operations
    arr = df.to_numpy()

    # Loop over all pairs of rows
    for i in range(arr.shape[0] - 1):
        for j in range(i + 1, arr.shape[0]):
            row1, row2 = arr[i, :], arr[j, :]
            
            # Find the indices where both rows have positive values
            common_cols = (row1 > 0) & (row2 > 0)
            
            if np.any(common_cols):  # If there is at least one common positive value
                # Check if there are "holes" (i.e., one row has a positive value, the other has zero)
                if np.any((row1 > 0) & (row2 == 0)) or np.any((row1 == 0) & (row2 > 0)):
                    return False  # If holes are found, it's not well-formed
    return True  # If no issues are found, it's well-formed

def check_valid(subset_df, valid_pairs):
    arr = subset_df.to_numpy()
    index = subset_df.index
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > 0:
                # for each pair of transtion <t1,t2> in M
                ordered_pair = (index[i], index[j])
                if ordered_pair not in valid_pairs:
                    return False
    return True


def compute_flex(self, cm):
        """
        The flex of a poset, or the coverage of poset:
            flex = 1 - cp/tp
        where cp is the number of comparable pairs of items in the poset, and tp is the total number of pairs.
        1 means totally unordered and 0 means totally ordered.
        """
        total_pairs = cm.shape[0] * (cm.shape[0] - 1) / 2
        comparable_pairs = np.count_nonzero(cm == 1)
        return 1 - comparable_pairs / total_pairs


def complete_by_transitivity(matrix: np.ndarray):
    """
    This function completes a partially ordered matrix by transitivity.
    It ensures that if A < B and B < C, then A < C.
    
    Args:
        matrix: A NumPy 2D array where np.nan represents incomparable pairs, and 1 represents comparable pairs.
    
    Returns:
        A NumPy 2D array with transitive relations completed.
    """

    n = matrix.shape[0]
    
    # Loop over each pair (i, j) in the upper triangle of the matrix
    for i in range(n):
        for j in range(i + 1, n):
            current = matrix[i, j]
            if current == 1:  # If the pair is comparable
                # Complete the transitivity for all elements between i and j
                for k in range(j + 1, n):  # Iterate over all elements after j
                    next_val = matrix[j, k]  # Check the relation between j and k
                    if next_val == 1:  # If the relation holds, propagate it
                        matrix[i, k] = 1                  
    return matrix

def get_to_comparable_matrix(trace):
        n = len(trace.steps)
        
        # Initialize a matrix of size n x n filled with np.nan
        cm = np.full((n, n), np.nan)
        
        # Set the upper triangular part (excluding the diagonal) to 1 (comparable pairs)
        for i in range(n):
            for j in range(i + 1, n):
                cm[i, j] = 1
        return cm

def get_po_comparable_matrix(input_dod, input_cm, strict):
    def destroy(gap, repeats):
        min_step = math.ceil(repeats/10)
        candidates = np.argwhere(output_cm == 1)
        np.random.shuffle(candidates)
        n = input_cm.shape[0]
        step = int(n*(n-1)/2 * gap)
        min_step = min(len(candidates), min_step)
        step = max(step,  min_step)
        idx_to_remove = np.random.choice(len(candidates), size=step, replace = False)
        for idx in idx_to_remove:
            x,y = candidates[idx]  
            output_cm[x,y] = np.nan
    if (input_dod ==0):
        return input_cm, 0
    if(input_dod == 1):
        output_cm = np.full_like(input_cm, np.nan)
        return output_cm, 1
    
    output_cm = input_cm.copy()


    if strict:
        repeats=0
        flag = True
        gap = input_dod
        while flag:
            dod = destroy(gap, repeats)
            output_cm = complete_by_transitivity(output_cm)
            dod = compute_flex(output_cm)
            repeats+=1
            if (dod >= input_dod):
                flag = False
            else:
                gap = input_dod - dod
                flag = True

    else:
        dod = destroy(input_dod, 1)

    return output_cm, dod

def get_potrace(cm, dod, to_trace, reorder):
    from traces import PartialOrderedTrace, PartialOrderedStep

    po_steps: List[PartialOrderedStep] = [
        PartialOrderedStep(to_step.state, to_step.action, to_step.index, []) for to_step in to_trace.steps
    ]
    for i in range(len(cm)):
        for j in range(i+1, len(cm)):
            if cm[i,j] == 1:
                po_steps[i].successors.append(j) 
    if reorder:
        random.shuffle(po_steps)
    
    return PartialOrderedTrace(po_steps, dod)

def convert_trace_to_potrace(to_trace, input_dod, strict=True, reorder=True):
    to_cm = get_to_comparable_matrix(to_trace)
    po_cm, output_dod = get_po_comparable_matrix(input_dod, to_cm, strict)
    return get_potrace(po_cm, output_dod, to_trace)
