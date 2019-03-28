"""Required modules."""
import csv
import re
import sys

import numpy as np
import scipy.io as sio
import sympy as sym
import xlrd

DATE = xlrd.XL_CELL_DATE
TEXT = xlrd.XL_CELL_TEXT
BLANK = xlrd.XL_CELL_BLANK
EMPTY = xlrd.XL_CELL_EMPTY
ERROR = xlrd.XL_CELL_ERROR
NUMBER = xlrd.XL_CELL_NUMBER


def read_excel(filename, sheet=None):
    """
    Read sheet data or sheet names from an Excel workbook into a :class:`Spreadsheet`.

    :example:

    sheet_names = read_excel('parameter.xlsx') # returns a list of sheet names

    :example:

    spreadsheet = read_excel('parameter.xlsx', 0) # read the first sheet

    :example:

    spreadsheet = read_excel(parameter.xls', 'sheet_2') # load 'sheet_2'

    :param filename: name of the excel woorkbook to import
    :param sheet: spreadsheet name or index to import
    :type filename: string
    :type sheet: string or integer or None
    :return: sheet names if sheet is None, otherwise sheet data
    :rtype: list of strings if sheet is None, otherwise :class:`Spreadsheet`
    """
    book = xlrd.open_workbook(filename)
    spreadsheet = Spreadsheet()
    if sheet is None:
        return book.sheet_names()
    elif isinstance(sheet, int):
        xl_sheet = book.sheet_by_index(sheet)
        spreadsheet.set_data(xl_sheet.get_rows())
        return spreadsheet
    else:
        xl_sheet = book.sheet_by_name(sheet)
        spreadsheet.set_data(xl_sheet.get_rows())
        return spreadsheet


def loadtxt(filename, dtype='float', comments='#', delimiter=None, skiprows=0,
            usecols=None, unpack=False):
    """Load ascii files into a numpy ndarray using numpy.loadtxt."""
    return np.loadtxt(
        filename, dtype, comments, delimiter,
        None, skiprows, usecols, unpack)


def load(file, mmap_mode=None, allow_pickle=True, fix_imports=True,
         encoding='ASCII'):
    """Load numpy .npy and .npz files to an array or map of arrays respectively using np.load."""
    return np.load(file, mmap_mode, allow_pickle, fix_imports, encoding)


def read_csv(filename, start=1, stop=None, assume=TEXT):
    """
    Read a csv file into a :class:`Spreadsheet`.

    :example:

    sheet = read_csv('parameters.csv', start=9, assume=NUMBER)

    :param filename: name of the file to read
    :param start: row to start reading
    :param stop: row to stop reading
    :param assume: type of data to assume
    :type filename: string
    :type start: integer
    :type stop: integer
    :type assume: integer
    :return: spreadsheet data
    :rtype: :class:`Spreadsheet`
    """
    values = []
    spreadsheet = Spreadsheet(assume)
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            values.append(row)

    if stop is None:
        stop = len(values)

    values = values[start - 1:stop]
    spreadsheet.set_values(values)
    return spreadsheet


def load_mat(filename, variable):
    """
    Read the variable from filename.

    :example:

    sheet = read_mat("parameter.mat", "cse")

    :param filename: name of the .mat file to read
    :param variable: variable to load
    :type filename: string
    :type variable: string
    :return: variable data
    :rtype: array
    """
    contents = sio.loadmat(filename)
    return contents[variable]


def load_section(sheet, row_range=None, col_range=None):
    """
    Read a 'chunk' of data from a spreadsheet.

    Given a selection of rows and columns, this function will return the
    intersection of the two ranges. Note that the minimum value for each range
    is 1.

    :example:

    spreadsheet = read_excel('parameters.xlsx', 'Parameters')
    cell_data = load_section(spreadsheet, [1, 3, 5], range(7, 42))

    :param sheet: spreadsheet data
    :param row_range: selected rows
    :param col_range: selected columns
    :type sheet: :class:`xlrd.sheet`
    :type row_range: list of integers or integer
    :type col_range: list of integers or integer
    :return: section of sheet data
    :rtype: array if assume=NUMBER else list
    """
    if row_range is None:
        row_range = range(1, len(sheet.values) + 1)

    if col_range is None:
        col_range = range(1, len(sheet.values[0]) + 1)

    if isinstance(row_range, int):
        row_range = [row_range]

    if isinstance(col_range, int):
        col_range = [col_range]

    rval = [[sheet.cell(x - 1, y - 1) for y in col_range] for x in row_range]

    if sheet.assume == NUMBER:
        return np.array(
            [[rval[x - 1][y - 1].value for y in col_range] for x in row_range], dtype='float')

    return rval


def _multiple_replace(repl, text):
    """
    Replace multiple regex expressions.

    :param repl: dictionary of values to replace
    :param text: text to perform regex on
    :type repl: dict
    :type text: string
    :return: processed text
    :rtype: string
    """
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, repl.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: repl[mo.string[mo.start():mo.end()]], text)


def _fun_to_lambda(entry):
    """
    Convert a given string representing a matlab anonymous function to a lambda function.

    :example:
    lambdafun = "@(x) cos(x)"
    lambdafun(np.pi)

    :param entry: string of matlab anonymous equation
    :type: string
    :return: mathmatical function
    :rtype: lambda function
    """
    repl = {
        './': '/',
        '.*': '*',
        '.^': '**'
    }

    # pull out function variable definition
    vari = re.findall(r'\@\(.*?\)', entry)
    vari = [re.sub(r'\@|\(|\)', '', x) for x in vari]

    # remove variable definition
    entry = re.sub(r'\@\(.*?\)', '', entry)

    # replace operators to suit numpy
    entry = _multiple_replace(repl, entry)

    # separate equations into different functions
    entry = re.sub('{|}', '', entry).split(',')

    return list(sym.sympify(entry[i]) for i in range(0, len(entry)))


def load_params(sheet, rows=None, ncols=None, pcols=None, cols=None,
                nrows=None, prows=None):
    """
    Read designated parameters from the sheet.

    :example:

    sheet=read_excel('parameter_list.xlsx', 0, 'index')
    params["pos"] = load_params(sheet, range(55, 75), ncols=2, pcols=3)

    :param sheet: spreadsheet data
    :type sheet: ldp.Spreadsheet
    :param rows: same as nrows=prows
    :type rows: range
    :param cols: same as ncols=pcols
    :type cols: range
    :param nrows: cell rows to read for parameter names
    :type nrows: int
    :param ncols: cell columns to read for parameter names
    :type ncols: int
    :param prows: cell rows to read for parameter data
    :type prows: int
    :param pcols: cell columns to read for parameter data
    :type pcols: int
    :return: mapping of parameter names to values
    :rtype: dict
    """
    if rows:
        nrows = rows
        prows = rows

    if cols:
        ncols = cols
        pcols = cols

    name_cells = load_section(sheet, nrows, ncols)
    data_cells = load_section(sheet, prows, pcols)

    # Verify the number of names matches the number of params
    assert len(name_cells) == len(data_cells)

    data = [_fun_to_lambda(x.value) if x.ctype == TEXT else
            x.value if x.ctype == NUMBER else None
            for y in data_cells for x in y]

    return dict(zip([x.value for y in name_cells for x in y], data))


class Spreadsheet(object):
    """Hold spreadsheet data."""

    def __init__(self, assumption=None):
        """Entry point for :class:`Spreadsheet`."""
        self.values = None
        self.ctypes = None
        self.assume = assumption

    def set_data(self, data_in):
        """Set spreadsheet data using cell generators."""
        data = list(data_in)
        self.values = [[col.value for col in row] for row in data]
        self.ctypes = [[col.ctype for col in row] for row in data]

    def set_values(self, values):
        """
        Set spreadsheet cell values.

        :param values: values to set
        :type values: container, e.g. list
        """
        self.values = values

    def set_ctypes(self, ctype):
        """
        Set spreadsheet cell types. I.e. NUMBER, TEXT, etc.

        :param ctype: cell types to set
        :type values: container, e.g. list
        """
        self.ctypes = ctype

    def size(self):
        """
        Retrieve the dimensions of the spreadsheet.

        :return: spreadsheed dimensions
        :rtype: tuple
        """
        if self.values is not None:
            return len(self.values), len(self.values[0])
        else:
            return None

    def cell(self, xpos, ypos):
        """
        Retrieve cell information.

        :param xpos: cell row
        :param ypos: cell column
        :type xpos: integer
        :type ypos: integer
        :return: cell values and info
        :rtype: :class:`xlrd.sheet.Cell`
        """
        if self.ctypes:
            return xlrd.sheet.Cell(
                self.ctypes[xpos][ypos], self.values[xpos][ypos])
        elif self.assume:
            return xlrd.sheet.Cell(self.assume, self.values[xpos][ypos])
        else:
            return None


def main():
    """
    Allow module to be run from the console.

    TODO: Add interface
    """
    pass


if __name__ == '__main__':
    sys.exit(main())
