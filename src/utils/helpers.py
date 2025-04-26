from IPython.display import display, HTML
from tabulate import tabulate
from typing import List, Set
import numpy as np

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
