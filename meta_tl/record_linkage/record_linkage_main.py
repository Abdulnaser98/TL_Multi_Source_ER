""" Main module for linking records

    This module calls the necessary modules to perform
    the funcionalities of the record linkage process

"""
import sys
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/record_linkage')
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')

# Import necessary modules
import os
import blocking
import comparison
import load_data_set
import utils as utl

path_to_working_dir = '/Users/abdulnaser/Desktop/TL_Multi_Source_ER/'
path_to_data_folder = path_to_working_dir + 'data/'
path_to_sim_vector_folder =  path_to_data_folder + 'linkage_tasks/'



def record_linkage_main():
    # Delete all files in the sim_vector_folder
    utl.delete_files(path_to_sim_vector_folder)

    # Header line available True or False
    header_line = True

    # The attribute number that contains the record identifier
    rec_id_col = 0

    # The list of attribuites to use for blocking.
    blocking_attr_list = ['famer_brand_list', 'famer_model_no_list']


    cleaned_data_folder_path = "/Users/abdulnaser/Desktop/TL_Multi_Source_ER/data/cleaned_data/"

    # List all CSV files in the folder
    csv_files = [ file for file in os.listdir(cleaned_data_folder_path) if file.endswith('.csv')]

    list_of_rec_dicts = []
    list_of_blocking_dicts = []
    #================================================ Blocking ===================================================
    for dataset in csv_files:
        print("The file to be preocessed is {}".format(dataset))
        dataset_path = os.path.join(cleaned_data_folder_path, dataset)
        rec_dict = load_data_set.load_data_set(dataset_path, rec_id_col, header_line)
        block_dict = blocking.blocking(rec_dict, blocking_attr_list)
        list_of_rec_dicts.append(rec_dict)
        list_of_blocking_dicts.append(block_dict)


    #================================================ comparison =================================================
    approx_comp_funct_list = [ (comparison.find_max_comparison_TruncateBegin20, 'famer_model_no_list', 'famer_model_no_list'),  # Modell-liste
                            (comparison.find_max_comparison_TruncateBegin20, 'famer_mpn_list', 'famer_mpn_list'),  # MPN-Liste
                            (comparison.find_max_comparison_TruncateBegin20, 'famer_ean_list', 'famer_ean_list'),  # EAN-Liste
                            (comparison.dice_comp, 'famer_product_name','famer_product_name'), # product-name,
                            (comparison.find_max_comparison_3gram, 'famer_model_list', 'famer_model_list'),
                            (comparison.NumMaxProz30, 'famer_digital_zoom', 'famer_digital_zoom'),  # digital-zoom
                            (comparison.NumMaxProz30, 'famer_optical_zoom', 'famer_optical_zoom'),  # optical-zoom
                            (comparison.NumMaxProz30, 'famer_width', 'famer_width'),  # Breite
                            (comparison.NumMaxProz30, 'famer_height', 'famer_height'),  # Hohe
                            (comparison.NumMaxProz30, 'famer_weight', 'famer_weight'),  # Gewicht
                            (comparison.jaccard_comp, 'famer_sensor', 'famer_sensor') # Sensor typ
                            #(comparison.NumMaxProz30, 'famer_resolution_from', 'famer_resolution_from'), # resoultion_from
                            #(comparison.NumMaxProz30, 'famer_resolution_to', 'famer_resolution_to'), # resoultion_to
                            #(comparison.NumMaxProz30, 'famer_megapixel', 'famer_megapixel') # Megapixel
                            ]

    processed_pairs = []
    for rec_dict_num1 in range(len(list_of_rec_dicts)):
        for rec_dict_num2 in range(len(list_of_rec_dicts)):
                source_1 = next(iter(list_of_rec_dicts[rec_dict_num1].values())).get('source')
                source_2 = next(iter(list_of_rec_dicts[rec_dict_num2].values())).get('source')
                if source_1 != source_2:
                    source1_source2 = source_1 + "_" + source_2
                    source2_source1 = source_2 + "_" + source_1
                    if (source1_source2 in processed_pairs) or (source2_source1 in processed_pairs):
                        pass
                    else:
                        sim_vec_dict = comparison.compareBlocks(list_of_blocking_dicts[rec_dict_num1], list_of_blocking_dicts[rec_dict_num2],
                                                                list_of_rec_dicts[rec_dict_num1], list_of_rec_dicts[rec_dict_num2],
                                                                approx_comp_funct_list,source1_source2)
                        processed_pairs.append(source1_source2)
                        processed_pairs.append(source2_source1)




record_linkage_main()


