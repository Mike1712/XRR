from pandas import read_csv
import os
import pickle
import PyPDF4
def prepend_line(file_name, line):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line)
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

def read_lines_file(file, onlyFirstLine=False):
    openfile = open(file, 'r')
    if onlyFirstLine:
        lines = openfile.readline()
    else:
        lines = openfile.readlines()
    openfile.close()
    return lines

def readEldensFile(file):
    lines = read_lines_file(file)
    counter = 0
    for i in range(len(lines)):
        len_lines = len(lines[i].split())
        if len_lines == 3:
            counter += 1
            if counter == 1:
                if i == 0:
                    header_len = 0
                else:
                    header_len = i-1
    colnames = ['z', 'delta', 'beta']   
    data = read_csv(file, header=header_len, sep='\s+', names = colnames, usecols = colnames)

    return data

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

def determine_headerlen_outfile(file):
    lines = read_lines_file(file)
    l = 0
    while l in range(0, len(lines)):
        if not "C" in lines[l]:
            headerlen = l - 1
            break
        l += 1
    return headerlen

def find_text_in_pdf(file, text2search:str):
    if not os.path.isfile(file):
        print(f'{f}\n is not a file.')
        return
    for f in files:
        if not any(x in f for x in ['blackened', 'Kopie']):
            pdf_obj = open(f, 'rb')
            pdf_reader = PyPDF4.PdfFileReader(pdf_obj)
            if pdf_reader.isEncrypted:
                pdf_reader.decrypt('')
            anz_pages = pdf_reader.numPages
            for i in range(anz_pages):
                page_obj = pdf_reader.getPage(i)
                text = page_obj.extractText()
                # pprint(text)
                if text2search in text:
                    print(f'Found {text2search} in\n\t{f}\n on page: {i}.')

def get_path_of_file_by_ext(path, subdirs = True, ext = '.dat', ignore_dirs = [], ignore_names = []):
    if subdirs:
        for root, dirs, files in (os.walk(path)):
            for d in dirs:
                if not any([x in d for x in ignore_dirs]) and os.path.isdir(os.path.join(root, d)):
                    for f in files:
                        print(os.path.join(root, d, f))
           # for name in dirs:
           #    print(os.path.join(root, name))

def save_data_to_pickle(path, obj, fname = None):
    '''
    Write data to pickle file
    -----------
    Parameters:
        * path: path_or_buf, filepath, where pickled data should be saved
        * fname: str, name of file without extension
    -----------
    Returns:
        pkl-file
    '''
    if not fname:
        print('no name given')
        name =  'pickle_file'
    else:
        name = fname
    pickle_file = os.path.join(path , f'{fname}.pkl')
    file_obj = open(pickle_file, 'wb')
    pickle.dump(obj, file_obj)
    file_obj.close()

def load_from_pickle(file):
    '''
    Load pickled data
    -----------
    Parameters:
        * file: absolute path of file
    ----------
    Returns: pickled data
    '''
    file_obj = open(file, 'rb')
    data = pickle.load(file_obj)
    file_obj.close()
    return data
