import os, sys
import re
import numpy as np
from pprint import pprint
from natsort import natsorted
from pandas import read_csv, DataFrame
from XRR.utilities.file_operations import read_lines_file
from shutil import copyfile, move as smove

def new_listdir(filepath):
    try:
        files = [os.path.join(filepath, f) for f in os.listdir(filepath)]
    except TypeError:
        files = []
        for f in os.listdir(filepath):
            file = os.path.join(filepath, f.decode('utf-8'))
            files.append(file)
    return files

def scan_numbers_arr(start_scan, anz_scans):
    end_scan_number = start_scan + (anz_scans - 1)
    scan_numbers = np.arange(start_scan, end_scan_number + 1, step=1)
    return scan_numbers

def replace_chars(string, replacements):
    for char, repl in replacements:
        if char in string:
            string = string.replace(char, repl)
    return string


def rename_files(path, new_basenames, skip_files = list(), basename_prefix = '', replace_whitspace_by = '_', rename = False, shownames = False, file_ext = '.dat'):
    
    # replacements = [('°C', 'C'), ('@', '_'), (',', ''), (' ', replace_whitspace_by), ('(',''), (')',''), ('.',''), ('#','numb')]
    replacements = [('°C', 'C'), ('@', '_'), (',', 'p'), (' ', replace_whitspace_by), ('(',''), (')',''), ('#','numb'), ('.','p'), ('__', '_'),('___','_') ]
    files = new_listdir(path)
    files = [f for f in files if not any(x in f for x in skip_files) and f.endswith(file_ext)]
    files = natsorted(files)
    basenames = [os.path.basename(f) for f in files]
    if not len(basenames) == len(new_basenames):
        print(len(basenames), basenames,'\n',len(new_basenames), new_basenames)
        print(f'length of inputfiles does not match length of new names.')
        sys.exit
    # elif not any(x in f for x in new_basenames for f in files):
    elif not all(x in f for x in new_basenames for f in files):
        if isinstance(basename_prefix, str):
            new_basenames = [basename_prefix + '_' + replace_chars(nb, replacements) if not basename_prefix == '' else replace_chars(nb, replacements) for nb in new_basenames]
        elif isinstance(basename_prefix, list):
            new_basenames = [bp + '_' + replace_chars(nb, replacements) if not bp == '' else replace_chars(nb, replacements) for nb, bp in zip(new_basenames, basename_prefix)]
        new_basenames = [replace_chars(nb, replacements) for nb in new_basenames]
        new_files = [os.path.join(path, nb) + file_ext for nb in new_basenames]

    else:
        # [print(f,x) for f, x in zip(files, new_basenames)]
        print('Already renamed files.')
        sys.exit
    if shownames:
        for i in range(len(files)):
            print(f'Old name: {os.path.basename(files[i])}, new name: {os.path.basename(new_files[i])}')
    if rename:
        [os.rename(files[i], new_files[i]) for i in range(len(files))]
        
        

def delete_neg_values(savepath, file_ext = '.dat', saveOriginalFiles = False, replaceWithZeroWeightVals = False):    
    '''
    Delte negative counts from a file. File at least has to contain two coloumns: incident angles and counts. If counts should be weighted, weights should be last column
    '''
    colnames = ['q', 'counts', 'weigths']
    all_dat_files_full_path = []
    for root, dirnames , f_names in os.walk(savepath):
        for f in f_names:
            if f.endswith(file_ext) and not any(x in f for x in ['orig', 'unweighted', 'unshifted']):
                oldname_fp=os.path.join(root,f)
                newname_fp = os.path.join(root,f).strip(file_ext) + '_orig' + file_ext
                firstline = read_lines_file(oldname_fp, onlyFirstLine=True).split()
                anz_cols = len(firstline)
                df = read_csv(oldname_fp, header=None,delimiter='\s+', names=colnames[0:anz_cols])
                if anz_cols < 3:
                    df[colnames[-1]] = np.repeat(1, len(df.q))
                try:
                    no_neg_values = df[df.counts >=10**(-12)]
                    neg_val_inds = df.index[df.counts <0]        
                    zeroWeightDf = df
                    zeroWeightDf.loc[neg_val_inds, ['counts', 'weigths']] = 1e-10, 0
                    if len(no_neg_values) < len(df):
                        if saveOriginalFiles:
                            smove(oldname_fp, newname_fp)
                        else:
                            os.remove(oldname_fp)
                        if not replaceWithZeroWeightVals:
                            no_neg_values.to_string(oldname_fp, index =False, header=False)
                        else:
                            zeroWeightDf.to_string(oldname_fp, index =False, header = False)
                        print(f' File: {os.path.basename(oldname_fp)} contained {len(df.q) - len(no_neg_values.q)} negative values.')
                        if saveOriginalFiles:
                            print(f'Old file with negative values saved as:\n{os.path.basename(newname_fp)}\nNew file saved as:\n{os.path.basename(oldname_fp)}')
                        else:
                            print(f'File {os.path.basename(oldname_fp)} does not contain negative values')
                except Exception as e:
                    print(f'File{os.path.basename(oldname_fp)} has not the correct format.')
                    print(e)
                    continue

def makedirlsfit(filepath, lsfitpath, ext = '.dat'):
    '''
    Create subdirectories in filepath and copy all files necessary to run lsfit into this subderictories. The filenames in the "*.par" and "LSFIT.OPT" files are changed automatically.
    ---------
    Parameters:
        * filepath: Path where the data files with q, intensity (, weights) are saved. 
        * lsfitpath: Path with lsfit template files. This path should contain:
            + LSFIT.OPT
            + reflek.exe
            + testcli.exe
            + pardat.par
            + condat.con
        * ext: str; Default ".dat". File ending of the datafiles.
    '''
    files = [f for f in new_listdir(filepath) if f.endswith(ext)]
    lsfitfiles = [lsf for lsf in new_listdir(lsfitpath)]
    subfolders = [f.split(ext)[0] for f in files]
    [print(sd) for sd in subfolders]
    for f, sd in zip(files, subfolders):
        try:
            filename = sd.split('/')[-1]
            confilename = os.path.join(sd, filename + '.con')
            parfilename = os.path.join(sd, filename + '.par')
            os.mkdir(sd)
            smove(f, sd)
        except:
            print('Subdirectories already created.')

        for lsfile in lsfitfiles:
            if lsfile.endswith('.con'):
                confile_template = lsfile 
                copyfile(confile_template, confilename)

            elif lsfile.endswith('.par'):
                parfile_template = lsfile
                copyfile(parfile_template, parfilename)
                with open(parfilename, 'r') as f:
                    pardata = f.readlines()
                    pardata[-1] = filename + ext
                with open(parfilename, 'w') as f:
                    f.writelines(pardata)
        
            elif lsfile.endswith('.OPT'):
                file =os.path.join(sd, lsfile.split('/')[-1])
                copyfile(lsfile, file)
                with open(file,'r') as f:
                    optdata = f.readlines()
                    optdata[0] = filename + '\n'
                with open(file,'w') as f:
                    f.writelines(optdata)

            else:
                file =os.path.join(sd, lsfile.split('/')[-1])
                copyfile(lsfile, file)


def printElementsFromMultipleLists(LoL:list, delim = ', '):
    anz_lists = len(LoL)
    lst_inds = [i for i in range(len(LoL))]

    # check if all lists have the same length
    lengths = map(len, LoL)
    if len(set(map(len,LoL)))==1:
        for sublist in zip(*LoL):
            print(sublist)
    else:
        print("Not all lists have the same lenght.")
        [print(len(l)) for l in LoL]
        return

def extract_values(dictionary, subkey):
    values = []
    for value in dictionary.values():
        if isinstance(value, dict):
            values.extend(extract_values(value, subkey))
        elif isinstance(value, (int, float)) and subkey in dictionary:
            values.append(dictionary[subkey])
    return values

def extractPressureVals(lst, round_precision = 2, air_pressure = 0):
    result_lst = list()
    for item in lst:
        if isinstance(item, (int, float)):
            result_lst.append(float(item))
        else:
            match = re.search(r'\d+(?:[.,]\d+)?', str(item))
            if match:
                value = float(match.group().replace(',', '.'))
                if any(n in item for n in ('Luft','luft', 'air', 'Air')):
                    value = air_pressure
            else:
                continue
        rounded_value = round(value, round_precision)
        result_lst.append(rounded_value)
    return result_lst

def find_closest_values_in_series(s, x):
    idx = (s - x).abs().idxmin()
    return idx, s.loc[idx]
