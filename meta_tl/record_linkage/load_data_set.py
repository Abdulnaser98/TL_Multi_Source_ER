""" Module with functionalities for reading data from a
    file and return a dictionary with record identifieres as
    keys and a list of attribute values
"""

# Import necessary modules

import csv
import gzip


def load_data_set(file_name, rec_id_col,header_line):
    """Load the data set and store in memory as a
       dictionary with record identifieres as keys.

       Parameter Description:
       file_name: Name of the data file to be read (CSV or CSV.GZ file)
       rec_id_col: Record identifier column of the data file
       header_line: Availability of the header line (True or False)
    """

    # Open a CSV file
    in_f = open(file_name)
    csv_reader = csv.reader(in_f)

    #print('Load data set from file: ' + file_name)

    if header_line:
        header_list = next(csv_reader)
        #print(' Header line: '+ str(header_list))
    #print(' Record identifier attributes: ' + str(header_list[rec_id_col]))
    #print('Attributes to use: ')
    #print(header_list)


    rec_num = 0
    rec_dict = {}

    rec_id_col = header_list.index('key')
    # Iterate through the record in the file
    for rec_list in csv_reader:
        rec_num += 1
        # Get the record identifier
        rec_id = rec_list[rec_id_col].strip().lower()

        rec_val_dict= {} # One value list per record

        for attr_id in range(len(rec_list)):
            #rec_val_list.append(rec_list[attr_id].strip().lower())
            rec_val_dict[header_list[attr_id]] = rec_list[attr_id].strip().lower()

        rec_dict[rec_id] = rec_val_dict

    in_f.close()


    if len(rec_dict) < rec_num:
        print(' *** Warning , data set contains %d duplicates ***' % (rec_num - len(rec_dict)))
        print(' %d unique records ' % (len(rec_dict)))



    # return the generated dictionary of records
    return rec_dict


