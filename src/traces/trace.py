from typing import List
from .step import Step
from .partialOrderedTrace import PartialOrderedTrace
import numpy as np
import random
import math



def compute_flex(cm):
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

def get_po_comparable_matrix(input_flex, input_cm, strict):
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
    if (input_flex ==0):
        return input_cm, 0
    if(input_flex == 1):
        output_cm = np.full_like(input_cm, np.nan)
        return output_cm, 1
    
    output_cm = input_cm.copy()


    if strict:
        repeats=0
        flag = True
        gap = input_flex
        while flag:
            flex = destroy(gap, repeats)
            output_cm = complete_by_transitivity(output_cm)
            flex = compute_flex(output_cm)
            repeats+=1
            if (flex >= input_flex):
                flag = False
            else:
                gap = input_flex - flex
                flag = True

    else:
        flex = destroy(input_flex, 1)

    return output_cm, flex

def get_potrace(cm, flex, to_trace, reorder):
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
    
    return PartialOrderedTrace(po_steps, flex)

def convert_trace_to_potrace(to_trace, input_flex, strict=True, reorder=True):
    to_cm = get_to_comparable_matrix(to_trace)
    po_cm, output_flex = get_po_comparable_matrix(input_flex, to_cm, strict)
    return get_potrace(po_cm, output_flex, to_trace, reorder)

class Trace:
    def __init__ (self, steps: List[Step]):
        self.steps = steps

    def __len__(self):
        return len(self.steps)
    
    def __iter__(self):
        return iter(self.steps)
    
    def __getitem__(self, index):
        return self.steps[index]
    
    def to_partial_ordered_trace(self, target_flex: float)-> PartialOrderedTrace:
        return convert_trace_to_potrace(self, target_flex, True, True)
    

