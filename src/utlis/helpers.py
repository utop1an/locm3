from IPython.display import display, HTML
from tabulate import tabulate

def pprint_table(matrix):
     display(tabulate(matrix, headers='keys', tablefmt='html'))