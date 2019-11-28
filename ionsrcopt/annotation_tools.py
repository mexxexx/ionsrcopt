import os
import re

import dateutil.parser as parser

import load_data as ld

def export_for_annotation(filename, column_to_export, rows):
    """ Export a section of time series data in a format that can be used for annotation with the tool from https://trainset.geocene.com/

    Parameters:
        filename (string): The name of the files, from where to extract the data
        column_to_export (string): Column to be extracted
        rows (list of range): Ranges of the rows to be extracted, all are going to be saved in seperate files in Data_Annotated/
    """
    filename_without_ext = os.path.splitext(os.path.basename(filename))[0]
    df = ld.read_data_from_csv(filename, ['Timestamp', column_to_export], None)
    df.insert(0, 'filename', 'f')
    df['label'] = 0
    df.columns = ['filename', 'timestamp', 'value', 'label']
    df.dropna(inplace=True)
    rows = [list(x) for x in rows]

    for x in rows:
        filename_output = filename_without_ext + "_" + str(x[0]).zfill(7) + "_" + str(x[-1]).zfill(7)
        print("Saving {}...".format(filename_output))

        sub_df = df.iloc[x].copy()
        sub_df['filename'] = filename_output
        sub_df['timestamp'] = sub_df['timestamp'].apply(lambda timestamp: parser.parse(timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
        sub_df.to_csv("Data_Annotated/" + filename_output + ".csv", index=False)

def get_annotated_files(dirname):
    """ List all .csv files in the folder 'Data_Annotated'

    Parameters:
        dirname (string): root directory of the files containing annotated data

    Returns:
        list of sting: A list of filenames
    """

    return [dirname + '/' + f for f in os.listdir(dirname + '/') if f.endswith('.csv')]

def import_annotated_data(filenames=[], dirname=''):
    """ Imports all the annotated data into dataframes

    Prameters: 
        filenames (list of string): The files to load. If empty, then everything from dirname will be imported
        dirname (string): Directory that should be loaded if no files are specified.

    Returns:
        list of DataFrame
    """

    if not filenames and not dirname:
        raise ValueError('At least one of the parameters has to be non empty')

    columns = ['timestamp', 'value', 'label']

    if not filenames:
        filenames = get_annotated_files(dirname)
    filenames.sort()

    annotated_dfs = []
    previous_groups = {}
    for f in filenames:
        groups = re.search(r"(.*)_([0-9]*)_([0-9]*).csv", f).groups()
        groups = { 'filename' : groups[0], 'start_index' : int(groups[1]) , 'end_index' : int(groups[2])  }
        df = ld.read_data_from_csv(f, columns, None)
        df = ld.convert_column(df, 'value', 'float32')
        df = ld.convert_column(df, 'label', 'int32')
        df.index = range(groups['start_index'], groups['end_index']+1)

        if previous_groups and previous_groups['filename'] == groups ['filename'] and previous_groups['end_index'] == (groups['start_index']-1):
            annotated_dfs[-1] =  annotated_dfs[-1].append(df)
        else:
            annotated_dfs.append(df)

        previous_groups = groups
    
    return annotated_dfs