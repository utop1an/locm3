from IPython.display import display, HTML
from tabulate import tabulate
from typing import List, Set

def pprint_table(matrix):
     display(tabulate(matrix, headers='keys', tablefmt='html'))

def pprint_list(lst: List):
     for item in lst:
        print(item, end=", ")
     print("")
