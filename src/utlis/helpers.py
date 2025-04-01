from IPython.display import display, HTML
from tabulate import tabulate
from typing import List, Set

def pprint_table(matrix):
     display(tabulate(matrix, headers='keys', tablefmt='html'))
