from IPython.display import display, HTML
from tabulate import tabulate
from typing import List, Set
from collections import defaultdict
import numpy as np
import math
import pandas as pd
import random

random.seed(42)

def pprint_table(matrix):
     display(tabulate(matrix, headers='keys', tablefmt='html'))

def pprint_list(lst: List):
     for item in lst:
        print(item, end=", ")
     print("")

def default_dict_factory(inner):
    return defaultdict(list)

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

def check_valid(subset_df, example_sequences):
    index = subset_df.index

    valid_example_sequences = [ [event for event in seq if event in index ] for seq in example_sequences]
    for seq in valid_example_sequences:
        for i in range(len(seq)-1):
            if subset_df.loc[seq[i], seq[i+1]] == 0:
                return False
    
    return True


def complete_PO(PO_matrix):
    changed = True
    while changed:
        changed = False
        for i in range(len(PO_matrix)):
            for j in range(len(PO_matrix)):
                if i==j:
                    continue
                current = PO_matrix.iloc[i,j]
                if (not pd.isna(current) and current == 1):

                    # complete matrix based on transitivity of PO
                    # if a>b, b>c, then a>c
                    for x in range(len(PO_matrix)):
                        if x==i or x ==j:
                            continue

                        next = PO_matrix.iloc[j,x]
                        if (next == 1 and PO_matrix.iloc[i,x] != 1):
                            PO_matrix.iloc[i,x] = 1
                            PO_matrix.iloc[x,i] = 0
                            changed = True

def complete_FO(FO_matrix, PO_matrix):
    for i in range(len(PO_matrix)):
        for j in range(len(PO_matrix)):
            if i==j:
                continue
            current_PO = PO_matrix.iloc[i,j]
            if current_PO == 0:
                FO_matrix.iloc[i,j] = 0
            elif current_PO == 1:
                flag=1
                for x in range(len(FO_matrix)):
                    if x != i and x !=j:
                        ix = PO_matrix.iloc[i,x]
                        xj = PO_matrix.iloc[x,j]
                        # not sure
                        if (pd.isna(ix)or pd.isna(xj)):
                            flag =2
                        # FO_ij should be 0
                        if ix==1 and xj==1:
                            flag=0
                            break
                # No change, FO_ij should be 1
                if flag==1:
                    FO_matrix.iloc[i,j]=1
                    FO_matrix.iloc[j,i] = 0
                    # check nans
                    for y in range(len(FO_matrix)):
                        if y!=i and y!=j:
                            FO_matrix.iloc[i,y] = 0
                            FO_matrix.iloc[y,j] = 0
                # FO_ij should be 0
                elif flag == 0:
                    FO_matrix.iloc[i,j]=0

def complete_PO_np(PO_matrix):
    n = PO_matrix.shape[0]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if i == j or PO_matrix[i, j] != 1:
                    continue
            
                    # complete matrix based on transitivity of PO
                    # if a>b, b>c, then a>c
                for x in range(n):
                    if x==i or x ==j:
                        continue

                    if PO_matrix[j, x] == 1 and PO_matrix[i, x] != 1:
                        PO_matrix[i, x] = 1
                        PO_matrix[x, i] = 0
                        changed = True

def complete_FO_np(FO_matrix, PO_matrix):
    n = PO_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            current_PO = PO_matrix[i,j]
            if current_PO == 0:
                FO_matrix[i,j] = 0
                continue
            if current_PO != 1:
                continue
            flag=1
            for x in range(n):
                if x == i or x == j:
                    continue
             
                ix = PO_matrix[i,x]
                xj = PO_matrix[x,j]
                if ix==1 and xj==1:
                    flag=0
                    break
                elif ix != ix or xj != xj:  # test for np.nan
                    flag = 2  # unsure
                # not sure
                if (pd.isna(ix) or pd.isna(xj)):
                    flag =2
                # FO_ij should be 0
                
            # No change, FO_ij should be 1
            if flag==1:
                FO_matrix[i,j]=1
                FO_matrix[j,i] = 0
                # check nans
                for y in range(n):
                    if y!=i and y!=j:
                        FO_matrix[i,y] = 0
                        FO_matrix[y,j] = 0
            # FO_ij should be 0
            elif flag == 0:
                FO_matrix[i,j]=0