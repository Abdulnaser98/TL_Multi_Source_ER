import math
import numpy as np
import pandas as pd


Q = 3  # Value length of q-grams for Jaccard and Dice comparison function

is_efficient = False
is_padding = True

def generate_trigrams(val):
    trigrams = []
    # Iterate through the text and generate trigrams
    for i in range(len(val) - 2):
        trigram = val[i:i+3]  # Extract three characters starting from index i
        trigrams.append(trigram)  # Add the trigram to the list
    return trigrams


def TruncateBegin20(val1,val2):

    first_20_chars_str1 = val1[:20]
    first_20_chars_str2 = val2[:20]

    # Compare the first 20 characters and return 1 if they are identical, 0 otherwise
    if first_20_chars_str1 == first_20_chars_str2:
        return 1
    else:
        return 0


def trigram_similarity(str1, str2):
    if (len(str1) < 3 or len(str2) < 3) and (len(str1) == len(str2)):
        if str1 == str2:
            return 1
        else:
            return 0

    elif (len(str1) < 3 or len(str2) < 3) and (len(str1) != len(str2)):
            return 0

    else:
        # Generate trigrams for both strings
        trigrams_str1 = generate_trigrams(str1)
        trigrams_str2 = generate_trigrams(str2)

        # Calculate the intersection of trigrams (common trigrams)
        common_trigrams = set(trigrams_str1) & set(trigrams_str2)

        # Calculate the Jaccard similarity coefficient
        similarity = len(common_trigrams) / len(set(trigrams_str1) | set(trigrams_str2))

        return similarity


def find_max_comparison_TruncateBegin20(str1, str2):
    max_value = 0

    # Split the strings using the backslash (\) delimiter
    list1 = str1.split(r'\,')
    list2 = str2.split(r'\,')

    # Remove leading and trailing whitespaces from each element in the lists
    list1 = [ item.strip() for item in list1]
    list2 = [ item.strip() for item in list2]

    # Iterate through all combinations of list elements
    for number_1 in list1:
        for number_2 in list2:
            # Compare the first 20 characters of the current combination
            comparison_result = TruncateBegin20(number_1, number_2)
            # Update max_value if the current comparison result is greater
            if comparison_result > max_value:
                max_value = comparison_result

    return max_value





def find_max_comparison_3gram(str1, str2):
    max_value = 0

    # Split the input strings using the backslah (\) delimiter
    list1 = str1.split(r'\,')
    list2 = str2.split(r'\,')

    # Remove leading and trailing whitespaces from each element in the lists
    list1 = [ item.strip() for item in list1]
    list2 = [ item.strip() for item in list2]

    # Iterate through all combinations of list elements
    for number_1 in list1:
        for number_2 in list2:
            # Compare the first 20 characters of the current combination
            comparison_result = trigram_similarity(number_1, number_2)
            # Update max_value if the current comparison result is greater
            if comparison_result > max_value:
                max_value = comparison_result

    return max_value
print("3 g is ")
print(find_max_comparison_3gram("g1","g1x"))


def dice_comp(val1, val2):
    """Calculate the Dice coefficient similarity between the two given attribute
     values by extracting sets of sub-strings (q-grams) of length q.

     Returns a value between 0.0 and 1.0.
  """
    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif val1 == val2:
        return 1.0
    if not is_efficient:
        if is_padding:
            pad_val_1 = "#" * (Q - 1) + val1 + "#" * (Q - 1)
            pad_val_2 = "#" * (Q - 1) + val2 + "#" * (Q - 1)
        else:
            pad_val_1 = val1
            pad_val_2 = val2
        q_gram_list1 = [pad_val_1[i:i + Q] for i in range(len(pad_val_1) - (Q - 1))]
        q_gram_list2 = [pad_val_2[i:i + Q] for i in range(len(pad_val_2) - (Q - 1))]
        q_gram_set1 = set(q_gram_list1)
        q_gram_set2 = set(q_gram_list2)
    else:
        q_gram_set1 = val1
        q_gram_set2 = val2
    if len(q_gram_set1) == 0 or len(q_gram_set2) == 0:
        return 0
    i = len(q_gram_set1.intersection(q_gram_set2))
    dice_sim = 2.0 * float(i) / float(len(q_gram_set1) + len(q_gram_set2))

    # ************ End of your Dice code ****************************************

    assert 0.0 <= dice_sim <= 1.0

    return dice_sim




def jaccard_comp(val1, val2):
    """Calculate the Jaccard similarity between the two given attribute values
     by extracting sets of sub-strings (q-grams) of length q.

     Returns a value between 0.0 and 1.0.
  """
    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement Jaccard similarity function here *********

    jacc_sim = 0.0  # Replace with your code
    if not is_efficient:
        if is_padding:
            pad_val_1 = "#" * (Q - 1) + val1 + "#" * (Q - 1)
            pad_val_2 = "#" * (Q - 1) + val2 + "#" * (Q - 1)
        else:
            pad_val_1 = val1
            pad_val_2 = val2
        q_gram_list1 = [pad_val_1[i:i + Q] for i in range(len(pad_val_1) - (Q - 1))]
        q_gram_list2 = [pad_val_2[i:i + Q] for i in range(len(pad_val_2) - (Q - 1))]

        q_gram_set1 = set(q_gram_list1)
        q_gram_set2 = set(q_gram_list2)
    else:
        q_gram_set1 = val1
        q_gram_set2 = val2

    i = len(q_gram_set1.intersection(q_gram_set2))
    u = len(q_gram_set1.union(q_gram_set2))
    assert u > 0, u

    jacc_sim = float(i) / u

    # ************ End of your Jaccard code *************************************

    assert jacc_sim >= 0.0 and jacc_sim <= 1.0

    return jacc_sim




def NumMaxProz30(num1, num2):

    # convert num1 and num2 to float if they are strings
    num1 = float(num1) if isinstance(num1,str) else num1
    num2 = float(num2) if isinstance(num2,str) else num2
    if num1 == 0 or num2 == 0:
        return '/'
    # Calculate the percentage difference between num1 and num2
    percentage_difference = abs(num1 - num2) / max(abs(num1), abs(num2)) * 100
    # If the percentage difference is 30% or more, return 0, otherwise scale the result between 0 and 1
    if percentage_difference >= 30:
        return 0
    else:
        # Scale the result between 0 and 1 based on the percentage difference
        similarity = 1 - (percentage_difference / 30)
        return similarity

# ================================================================
# Function to compare a block

def compareBlocks(blockA_dict, blockB_dict, recA_dict, recB_dict, attr_comp_list,file_name):
    """Build a similarity dictionary with pair of records from the two given
     block dictionaries. Candidate pairs are generated by pairing each record
     in a given block from data set A with all the records in the same block
     from dataset B.

     For each candidate pair a similarity vector is computed by comparing
     attribute values with the specified comparison method.

     Parameter Description:
       blockA_dict    : Dictionary of blocks from dataset A
       blockB_dict    : Dictionary of blocks from dataset B
       recA_dict      : Dictionary of records from dataset A
       recB_dict      : Dictionary of records from dataset B
       attr_comp_list : List of comparison methods for comparing individual
                        attribute values. This needs to be a list of tuples
                        where each tuple contains: (comparison function,
                        attribute number in record A, attribute number in
                        record B).

     This method returns a similarity vector with one similarity value per
     compared record pair.

     Example: sim_vec_dict = {(recA1,recB1) = [1.0,0.0,0.5, ...],
                              (recA1,recB5) = [0.9,0.4,1.0, ...],
                               ...
                             }
    """

    print('Compare %d blocks from dataset A with %d blocks from dataset B' % \
          (len(blockA_dict), len(blockB_dict)))

    print("The sources that we are comparing are ")
    print(file_name)
    sim_vec_dict = {} # A dictionary where keys are record pairs and values
    # lists of similatiry values


    # Iterate through each block in dictionary from dataset A
    for(block_bkv, rec_idA_list) in blockA_dict.items():
        # Check if the same blocking key occurs also for dataset B
        if(block_bkv in blockB_dict):
            # If so get the record identifier list from dataset B
            rec_idB_list = blockB_dict[block_bkv]

            # Compare each record in rec_id_listA with each record from rec_id_listB
            for rec_idA in rec_idA_list:
                recA = recA_dict[rec_idA]

                for rec_idB in rec_idB_list:
                    recB = recB_dict[rec_idB]

                    # generate the similiarty vector
                    sim_vec = compareRecord(recA, recB, attr_comp_list)

                    # Add the similarity vector of the compared pair to the similarity
                    # vector dictionary
                    #
                    sim_vec_dict[(rec_idA, rec_idB)] = sim_vec
    print('  Compared %d record pairs' % (len(sim_vec_dict)))
    print('')

    # save the sim vects as dataframe
    save_similiarty_vectors_into_dataframe(sim_vec_dict,file_name)

    return sim_vec_dict



def compareRecord(recA, recB, attr_comp_list):
    """Generate the similarity vector for the given record pair by comparing
     attribute values according to the comparison function and attribute
     numbers in the given attribute comparison list.

     Parameter Description:
       recA           : List of first record values for comparison
       recB           : List of second record values for comparison
       attr_comp_list : List of comparison methods for comparing attributes,
                        this needs to be a list of tuples where each tuple
                        contains: (comparison function, attribute number in
                        record A, attribute number in record B).

     This method returns a similarity vector with one value for each compared
     attribute.
  """
    sim_vec = []
    # Calculate a similarity for each attribute to be comapred
    #
    for (comp_funct, attr_numA, attr_numB) in attr_comp_list:
        #print("The attributes to compare are :")
        #print(attr_numA)
        #print(attr_numB)
        a_value = recA[attr_numA]
        b_value = recB[attr_numB]

        if a_value == '/':
            valA = '/'

        elif a_value == '' or pd.isnull(a_value):
            valA = '/' # Handle empty string or NaN values

        else:
            valA = recA[attr_numA]


        if b_value == '/':
            valB = '/'

        elif b_value == '' or pd.isnull(b_value):
            valB = '/' # Hanlde empty string or NaN values

        else:
            valB = recB[attr_numB]

        if valA == '/' or valB == '/':
            sim = '/'

        else:
            sim = comp_funct(valA, valB)

        sim_vec.append(sim)

    return sim_vec


def save_similiarty_vectors_into_dataframe(sim_vectors,file_name):

    # Extract keys and values from the dictionray
    keys = list(sim_vectors.keys())
    values = list(sim_vectors.values())


    # create lists to store data for DataFrame columns
    record_1 = [key[0] for key in keys]
    record_2 = [key[1] for key in keys ]

    # Create a dictionary for DataFrame creation
    data = {
        "record_compared_1": record_1,
        "record_compared_2": record_2,
        "Modell_no_Liste_TruncateBegin20": [val[0] for val in values],
        "MPN_Liste_TruncateBegin20": [val[1] for val in values],
        "EAN_Liste_TruncateBegin20": [val[2] for val in values],
        "Produktname_dic3": [val[3] for val in values],
        "Modell_Liste_3g": [val[4] for val in values],
        "Digital_zoom_NumMaxProz30": [val[5] for val in values],
        "optischer_zoom_NumMaxProz30": [val[6] for val in values],
        "Breite_NumMaxProz30": [val[7] for val in values],
        "Höhe_NumMaxProz30": [val[8] for val in values],
        "Gewicht_NumMaxProz30": [val[9] for val in values],
        "Sensortyp_Jaccard3": [val[10] for val in values]
    }

    # create DataFrame
    df = pd.DataFrame(data)

    # Specify the file path where you want to save the CSV file
    file_path = "/Users/abdulnaser/Desktop/TL_Multi_Source_ER/data/linkage_tasks/" + file_name + ".csv"

    # Save the DataFrame to the specified path as a CSV file
    df.to_csv(file_path, index=False)