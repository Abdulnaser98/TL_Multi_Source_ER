"""
Mdoule with functionalities for blocking on a
dictionary of records, where a blocking function
must return a dictionary with block idetifiera as
keys and values being sets or lists of record
identifiers in that block.

"""

import re
import pandas as pd



def blocking(rec_dict, blk_attr_list):
    """ Build the blocking index data structure
        (dictionary) to store blocking key values (BKY)
        as keys and the corresponding list of record
        identifieres.

        A blocking is implemented that simply
        concatenates attribute values.

        Parameter Description:
        rec_dict: Dictionary that holds the record
        identifieres as keys and corresponding list of
        record values

        blk_attr_list: List of blocking key attributes to use.

        This method returns a dictionary with blocking
        key values as its keys and list of record
        identifiers as its values (one list for each block).

        Example:
        If the blocking is based on 'postcode' then:
        block_dict = {'2000': [rec1_id, rec2_id, rec3_id, .....],
                      '2600': [rec4_id, rec5_id,..........]
                      .....
        }

        while if the blocking is based on 'postcode' and 'gender' then:
        block_dict = {'2000f': [rec1_id, rec3_id, ......],
                      '2000m': [rec2_id, ......]
                      '2600m': [rec4_id,.......]
                      ...
        }

    """

    # The dictionary with the blocks to be generated and returned
    block_dict = {}

    #print('Run blocking:')
    #print('List of blocking key attributes: ' + str(blk_attr_list))
    #print('Number of records to be blcoked: ') + str(len(rec_dict))
    #print('')

    for (rec_id , rec_values) in rec_dict.items():

        # Initialise the blocking key values for this record
        rec_bkv = ''

        # Process selected blocking attributes
        for attr in blk_attr_list:

            attr_val = rec_values[attr]
            if attr == 'famer_brand_list':
               rec_bkv += attr_val[:3]
            else:
                rec_bkv +=  attr_val[:100]

        # Insert the blocking key value and the record into
        # the blocking dictionary
        if (rec_bkv in block_dict): # Block key value in block index
            # Only need to add the record
            rec_id_list = block_dict[rec_bkv]
            rec_id_list.append(rec_id)

        else: # Block key value not in block index
            # Create a new block and add the record
            # identifier
            rec_id_list = [rec_id]
        block_dict[rec_bkv] = rec_id_list # store the new block

    return block_dict



def printBlockStatistics(BlockA_dict,dataset_name):
    """
     Calculate and print some basics statistics about
     the generated blocks
    """

    #print('Statistics of the generated blocks for {} : '.format(dataset_name))


    numA_blocks = len(BlockA_dict)


    block_sizeA_list = []
    for rec_id_list in BlockA_dict.values(): # Loop over all blocks
        block_sizeA_list.append(len(rec_id_list))

    print('Dataset A number of blocks generated: %d' % (numA_blocks))
    print(' Minimum block size: %d' % (min(block_sizeA_list)))
    print(' Average block size: %.2f' % \
          (float(sum(block_sizeA_list)) / len(block_sizeA_list)))
    print(' maximum block size: %d' % (max(block_sizeA_list)))
    print('')





