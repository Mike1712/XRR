from pandas import read_csv
def readDatFileData(data_file):
    '''
    Read data from file containing with q/counts/weights data (or similar data).
    Parameters:
    -----------
        data_file: Full path of file containing data.

    Returns:
    --------
        Data in datfile in pandas.DataFrame
    '''
    colnames = ['q', 'counts', 'weights']
    anz_cols = len(read_lines_file(data_file, onlyFirstLine=True).split())
    colnames = colnames[0:anz_cols]
    datfileData = read_csv(data_file, header = None, sep = '\s+', names = colnames)
    return datfileData

def read_lines_file(file, onlyFirstLine=False):
    openfile = open(file, 'r')
    if onlyFirstLine:
        lines = openfile.readline()
    else:
        lines = openfile.readlines()
    openfile.close()
    return lines