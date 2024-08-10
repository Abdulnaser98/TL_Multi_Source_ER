import re
import warnings
import numpy as np
import pandas as pd
#from utils import *
import sys
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
import config as conf
from utils import *

# Set the warning filter to "ignore"
warnings.filterwarnings("ignore")


columns_to_order = ['key', 'source','famer_model_list','famer_model_no_list','famer_mpn_list','famer_ean_list',
                    'famer_product_name', 'famer_digital_zoom','famer_optical_zoom', 'famer_width', 'famer_height',
                    'famer_weight', 'famer_sensor','famer_brand_list','famer_resolution_from','famer_resolution_to','famer_megapixel'
                   ]


path_to_raw_data = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/data/raw_data/'
path_to_cleaned_data = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/data/cleaned_data/'
path_to_help_data = '/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/data/help_data/'

# Define cleaning functions for each source with their specific important features
def clean_priceme_data(data):
    data = ensure_columns_exist(data, columns_to_order)
    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'priceme_data_cleaned.csv', index=False)



def clean_buy_data(data):
    # Cleaning logic for buy.net data using specific important features
    numeric_columns = ['digital zoom', 'optical zoom', 'height', 'width']
    data = clean_numeric_columns(data, numeric_columns)

    # Convert weight from grams to pounds
    data['weight'] = data['weight'].apply(convert_weight_to_grams)
    # Remove "g" from the 'weight' column
    data['weight'] = data['weight'].str.replace('g', '')

    buy_data_new_columns_names = {
    'weight': 'famer_weight',
    'digital zoom': 'famer_digital_zoom',
    'optical zoom': 'famer_optical_zoom',
    'height': 'famer_height',
    'width': 'famer_width',
    'image sensor': 'famer_sensor'
    }

    data.rename(columns=buy_data_new_columns_names, inplace=True)

    data['famer_height'] = data['famer_height'] * 2.54
    data['famer_width'] = data['famer_width'] * 2.54

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)
    # Save cleaned buy data to CSV
    data.to_csv(path_to_cleaned_data + 'buy_data_cleaned.csv', index=False)



def clean_buzzillions_data(data):
    # Clean numeric columns
    numeric_columns = ['optical zoom', 'digital zoom', 'width', 'height', 'famer_megapixel']
    data = clean_numeric_columns(data, numeric_columns)

    buzzillions_data_new_column_names = {
       'optical zoom': 'famer_optical_zoom',
       'digital zoom': 'famer_digital_zoom',
       'height': 'famer_height',
       'width' : 'famer_width',
       'sensor': 'famer_sensor'
   }

    data.rename(columns=buzzillions_data_new_column_names, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]
    # Save cleaned buzzillions data to CSV
    data.to_csv(path_to_cleaned_data + 'buzzillions_data_cleaned.csv', index=False)




def clean_eglobalcentral_data(data):

    # notices:
    # Width in cm
    # height in cm

    # Clean numeric columns
    numeric_columns = ['optical zoom', 'digital zoom']
    clean_numeric_columns(data, numeric_columns)

    # Consolidate weight and dimensions columns
    data['weight'] = data.apply(consolidate_weight_columns, axis=1)
    data['dimensions'] = data.apply(consolidate_dimension_columns, axis=1)

    #eglobalcentral_data_cleaned['dimensions'] = eglobalcentral_data_cleaned['dimensions'].str.replace(r'\([^)]*\)', '', regex=True)
    #eglobalcentral_data_cleaned['dimensions'] = eglobalcentral_data_cleaned['dimensions'].str.replace(r'\([^)]*\)', '', regex=True)

    # Extract the width and height dimensions
    data['extracted_width'] = data['dimensions'].apply(lambda x: extract_dimension(x, 'width',0))
    data['extracted_height'] = data['dimensions'].apply(lambda x: extract_dimension(x, 'height',1))

    # Correct some spellings in the Width and height columns
    data.loc[data['famer_product_name'] == 'panasonichcmdh2', 'extracted_width'] = np.nan
    data.loc[data['famer_product_name'] == 'panasonichcmdh2', 'extracted_height'] = np.nan


    # Convert the dimensions from mm to cm
    data['extracted_width'] = data['extracted_width'].str.replace('mm','')
    data['extracted_width'] = data['extracted_width'].apply(convert_mm_to_cm)

    data['extracted_height'] = data['extracted_height'].str.replace('mm','')
    data['extracted_height'] = data['extracted_height'].apply(convert_mm_to_cm)

    eglobalcentral_data_new_columns_names = {
        'digital zoom': 'famer_digital_zoom',
        'optical zoom': 'famer_optical_zoom',
        'weight': 'famer_weight',
        'extracted_width': 'famer_width',
        'extracted_height': 'famer_height',
        'Auflösung_x': 'famer_Auflösung_x',
        'Auflösung_y': 'fmaer_Auflösung_y',
        'famer_sensortype':'famer_sensor'
    }

    data.rename(columns=eglobalcentral_data_new_columns_names, inplace=True)

    print("The avaliable columns now are {} ".format(data.columns))
    # Drop some unnecessary columns
    #columns_to_drop = ['dimensions w x h x d', 'dimensions','famer_Auflösung_x','fmaer_Auflösung_y','famer_resolution_to','famer_resolution_from']
    #data.drop(columns=columns_to_drop, axis=1, inplace=True)


    # Check and create the missing columns
    data = ensure_columns_exist(data, columns_to_order)

    # Reorder the columns in the right way
    data = data[columns_to_order]

    # Save cleaned data to CSV
    data.to_csv(path_to_cleaned_data + 'eglobalcentral_data_cleaned.csv', index=False)



def clean_price_hunt_data(data):

    # Notices:
    # Width in cm
    # Height in cm
    # Weight in g


    # read a dataframe that contains manually cleaned infos about the weights
    cleaned_weight_dataframe = pd.read_csv(path_to_help_data + 'price_hunt_cleaned_weight_cleaned.csv', sep=";")

    # Merge dataframes on the common column
    all_relevant_data = pd.merge(data, cleaned_weight_dataframe[['weight_with_battery', 'weight_without_battery','famer_product_name']], on='famer_product_name', how='left')


    # Extract the dimensions_values from the dimensions column
    all_relevant_data['extracted_dimension_values'] = all_relevant_data['dimensions'].apply(extract_dimensions_values)


    # Extract the width, height and depth values from the dimensions values
    all_relevant_data['w_value'] = all_relevant_data['extracted_dimension_values'].apply(lambda x: x.get('w') if isinstance(x, dict) else None)
    all_relevant_data['h_value'] = all_relevant_data['extracted_dimension_values'].apply(lambda x: x.get('h') if isinstance(x, dict) else None)
    all_relevant_data['d_value'] = all_relevant_data['extracted_dimension_values'].apply(lambda x: x.get('d') if isinstance(x, dict) else None)

    # Convert the width and height values into cm
    all_relevant_data['w_value'] = all_relevant_data['w_value'].apply(convert_mm_to_cm)
    all_relevant_data['h_value'] = all_relevant_data['h_value'].apply(convert_mm_to_cm)

    # Fill NaN values in the "weight" column
    all_relevant_data['weight_with_battery'] = all_relevant_data['weight_with_battery'].fillna(all_relevant_data['weight_without_battery'])


    # Format the width , height and weight values to two decimal places
    all_relevant_data['w_value'] =  round(all_relevant_data['w_value'],2)
    all_relevant_data['h_value'] = round(all_relevant_data['h_value'],2)
    all_relevant_data['weight_with_battery'] = round(all_relevant_data['weight_with_battery'],2)


    #price_hunt_data_cleaned['famer_model_no_list'] = price_hunt_data_cleaned['famer_model_list'].str.replace(r'\D', '', regex=True)


    price_hunt_data_new_columns_names = {
        'digital zoom': 'famer_digital_zoom',
        'optical zoom': 'famer_optical_zoom',
        'weight_with_battery': 'famer_weight',
        'weight_without_battery':'famer_weight_without_battery',
        'w_value': 'famer_width',
        'h_value': 'famer_height',
        'sensor type': 'famer_sensor',
    }
    # Rename the relevant columns
    all_relevant_data.rename(columns = price_hunt_data_new_columns_names, inplace=True)


    # Extract the values of the digital and optical zoom
    all_relevant_data['famer_digital_zoom'] = all_relevant_data['famer_digital_zoom'].str.extract(r'([\d.]+)\s*x')
    all_relevant_data['famer_optical_zoom'] = all_relevant_data['famer_optical_zoom'].str.extract(r'([\d.]+)\s*x')

    # Create a new column 'row_number' representing the row numbers
    #price_hunt_data_cleaned.insert(0, 'row_key', price_hunt_data_cleaned.reset_index().index + 1)

    # Check and create missing columns
    all_relevant_data = ensure_columns_exist(all_relevant_data, columns_to_order)

    all_relevant_data = all_relevant_data[columns_to_order]
    all_relevant_data.to_csv(path_to_cleaned_data + 'price_hunt_cleaned.csv')


def clean_shopbot_data(data):

    # Extract dimensions (width, height, depth) from the 'dimensions' column
    data['width'] = data['dimensions'].apply(lambda x: extract_dimension(x, 'width',0))
    data['height'] = data['dimensions'].apply(lambda x: extract_dimension(x, 'height',1))
    data['depth'] = data['dimensions'].apply(lambda x: extract_dimension(x, 'depth',2))

    # Clean extracted dimension columns
    dimension_columns = ['width', 'height', 'depth', 'zoom', 'weight']
    data[dimension_columns] = data[dimension_columns].replace(r'[^0-9.]', '', regex=True)

    # Convert the dimension values from mm to cm
    data['width'] = data['width'].apply(convert_mm_to_cm)
    data['height'] = data['height'].apply(convert_mm_to_cm)

    shopbot_new_column_names = {
        'weight': 'famer_weight',
        'width': 'famer_width',
        'height': 'famer_height',
        'weight': 'famer_weight',
        'zoom': 'famer_optical_zoom',
    }
    data.rename(columns=shopbot_new_column_names, inplace=True)

    # List of column names to drop
    columns_to_drop = ['dimensions', 'video resolution', 'depth']

    # Drop the specified columns
    data.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]
    # Save cleaned data to CSV
    data.to_csv(path_to_cleaned_data + 'shopbot_data_cleaned.csv', index=False)



def clean_wexphotographic_data(data):
    # Extract dimensions (width, height, depth) from the 'size' column
    data['width'] = data['size'].apply(lambda x: extract_dimension(x, 'width',0))
    data['height'] = data['size'].apply(lambda x: extract_dimension(x, 'height',1))
    data['depth'] = data['size'].apply(lambda x: extract_dimension(x, 'depth',2))

    # Convert dimensions from mm to cm
    data['width'] = data['width'].apply(convert_mm_to_cm)
    data['height'] = data['height'].apply(convert_mm_to_cm)

    # Clean extracted numeric columns
    dimension_columns = ['width', 'height','weight g']
    data = clean_numeric_columns(data, dimension_columns)


    wexphotographic_data_new_column_names = {
        'optical zoom x': 'famer_optical_zoom',
        'weight g': 'famer_weight',
        'sensor type': 'famer_sensor',
        'width': 'famer_width',
        'height': 'famer_height'
    }

    data.rename(columns=wexphotographic_data_new_column_names, inplace=True)

    # List of column names to drop
    columns_to_drop = ['size', 'resolution', 'depth']

    # Drop the specified columns
    data.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Round numeric columns to two decimal digits after comma
    data['famer_width'] =  round(data['famer_width'],2)
    data['famer_height'] = round(data['famer_height'],2)
    data['famer_weight'] = round(data['famer_weight'],2)



    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    # Reorder the columns
    data = data[columns_to_order]
    # Save cleaned data to CSV
    data.to_csv(path_to_cleaned_data + 'wexphotographic_data_cleaned.csv', index=False)



def clean_garricks_data(data):
    garricks_data_new_column_names = {
    'weight': 'famer_weight',
    'sensor details': 'famer_sensor'
    }

    data.rename(columns=garricks_data_new_column_names, inplace=True)

    # List of column names to drop
    columns_to_drop = ['zoom range', 'resolution']

    # Drop the specified columns
    data.drop(columns=columns_to_drop, axis=1, inplace=True)

    data['famer_weight'] = data['famer_weight'].str.extract(r'([\d.]+)\s*g')

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'garricks_data_cleaned.csv',index=False)




def clean_pcconnection_data(data):
    data['extracted_values'] = data['physical dimensions'].apply(extract_dimensions_values)

    # Extract values for 'd,' 'w,' and 'h' from the extracted dictionary
    #data['d_value'] = data['extracted_values'].apply(lambda x: x.get('d') if isinstance(x, dict) else None)
    data['w_value'] = data['extracted_values'].apply(lambda x: x.get('w') if isinstance(x, dict) else None)
    data['h_value'] = data['extracted_values'].apply(lambda x: x.get('h') if isinstance(x, dict) else None)

    data['actual weight'] = data['actual weight'].apply(convert_weight_to_grams)


    pcconnection_data_new_column_names = {
        'digital zoom': 'famer_digital_zoom',
        'optical zoom': 'famer_optical_zoom',
        'optical sensor type': 'famer_sensor',
        'actual weight': 'famer_weight',
        'h_value': 'famer_height',
        'w_value': 'famer_width'
    }

    data.rename(columns=pcconnection_data_new_column_names, inplace=True)


    # List of column names to drop
    #columns_to_drop = ['physical dimensions', 'd_value','extracted_values']

    # Drop the specified columns
    #data.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]

    data['famer_digital_zoom'] = data['famer_digital_zoom'].str.replace('X','')
    data['famer_optical_zoom'] = data['famer_optical_zoom'].str.replace('X','')


    data.loc[data['famer_product_name']=='canoneos70d', 'famer_optical_zoom'] = np.nan
    data.to_csv(path_to_cleaned_data + 'pcconnection_data_cleaned.csv')




def clean_cammarkt_data(data):
    NOTES = "remove duplicate column Names"
    # Extract weight values before the first slash using regular expression
    data['extracted_weight'] = data['weight'].str.split('\\', n=1).str[0]
    # Remove "[" and double qoute at the beginning of the strings
    data['extracted_weight'] = data['extracted_weight'].str.replace(r'^\["', '', regex=True)
    data.loc[data['extracted_weight'] == '1', 'extracted_weight'] = data['weight'].str.split(',').str[1].str.replace(r'[\"\']', '').str.strip()
    data['extracted_weight'] = data['extracted_weight'].apply(convert_weight_to_grams)

    #width
    data['extracted_width'] = data['width'].str.split('\\', n=1).str[0]
    data['extracted_width'] = data['extracted_width'].str.replace(r'^\["', '', regex=True)
    data['extracted_width'] = data['extracted_width'].apply(convert_dimension_to_cm)
    data.loc[data['famer_product_name'] == 'olympusevolte620', 'extracted_width'] = 13
    data.loc[data['famer_product_name'] == 'nikon1j2', 'extracted_width'] = 10.6

    #height
    data['extracted_height'] = data['height'].str.split('\\', n=1).str[0]
    data['extracted_height'] = data['extracted_height'].str.replace(r'^\["', '', regex=True)
    data['extracted_height'] = data['extracted_height'].apply(convert_dimension_to_cm)

    #optical zoom
    data['extracted_optical_zoom'] = data['famer_opticalzoom']

    # digital zoom
    data['extracted_digital_zoom'] = data['digital zoom'].str.extract(r'(\d+)x')

    # sensor type
    data['extracted_sensor_typ'] = data['image sensor type'].str.split('\\', n=1).str[0].str.strip()

    # Columns to be removed
    columns_to_be_removed = ['famer_width','famer_height', 'camera resolution', 'width','height','optical zoom', 'image sensor type',
                             'camera resolution', 'image resolutions', 'optical zoom','digital zoom','weight','famer_opticalzoom','famer_weight',
                             'megapixels']

    # Remove the specified columns
    data.drop(columns= columns_to_be_removed, axis=1, inplace=True)


    cammarkt_data_new_column_names = {
        'extracted_weight': 'famer_weight',
        'extracted_width': 'famer_width',
        'extracted_height': 'famer_height',
        'extracted_optical_zoom': 'famer_optical_zoom',
        'extracted_digital_zoom': 'famer_digital_zoom',
        'extracted_sensor_typ': 'famer_sensor'
    }

    data.rename(columns=cammarkt_data_new_column_names, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'cammarkt_data_cleaned.csv')



def clean_gosale_data(data):

    data['famer_weight'] = data['famer_weight'] * 453.592
    gosale_data_new_column_names = {
        'famer_opticalzoom': 'famer_optical_zoom'
    }

    data.rename(columns=gosale_data_new_column_names, inplace=True)

    # Remove the specified columns
    data.drop(columns= ['product number mpn'], axis=1, inplace=True)


    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'gosale_data_cleaned.csv')



def clean_henrys_data(data):
    data['famer_opticalzoom'] = np.where(data['famer_product_name']== 'fujifilmxf1', 4, data['famer_opticalzoom'])

    # width
    data['width'] = data['dimensions wxhxd'].apply(lambda x: extract_dimension(x,'width',0))
    data['width'] = data['width'].str.replace('mm',' mm')
    data.loc[data['famer_product_name'] == 'panasonichxa500h', 'width'] = 26.5
    data.loc[data['famer_product_name'] == 'samsungwb50f', 'width'] = np.nan
    data['width'] = data['width'].apply(convert_mm_to_cm)
    #height
    data['height'] = data['dimensions wxhxd'].apply(lambda x: extract_dimension(x,'height',1))
    data['height'] = data['height'].str.replace('mm',' mm')
    data.loc[data['famer_product_name'] == 'panasonichxa500h', 'height'] = 68.5
    data.loc[data['famer_product_name'] == 'samsungwb50f', 'height'] = np.nan
    data['height'] = data['height'].apply(convert_mm_to_cm)

    # weight
    data['extracted_weight'] = data['weight'].str.extract(r'([\d.]+)\s*g')
    # digital zoom
    data['digital zoom'] = data['digital zoom'].str.replace('x','')
    # sensor type
    data['sensor type'] = data['sensor typesize'].apply(assign_sensor_type)


    # Columns to be removed
    #columns_to_be_removed = ['weight','dimensions wxhxd', 'sensor typesize', 'video resolution','famer_weight','optical zoom']

    # Remove the specified columns
    #henrys_data_non_null_columns.drop(columns= columns_to_be_removed, axis=1, inplace=True)



    henrys_data_new_column_names = {
        'digital zoom': 'famer_digital_zoom',
        'width': 'famer_width',
        'height': 'famer_height',
        'extracted_weight': 'famer_weight',
        'sensor type': 'famer_sensor',
        'famer_opticalzoom': 'famer_optical_zoom'
    }

    data.rename(columns=henrys_data_new_column_names, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]

    data['famer_digital_zoom'] = data['famer_digital_zoom'].replace('ma. 5',5)
    data.loc[data['famer_product_name']=='sigmadp3', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'fujifilmxf1', 'famer_digital_zoom'] = 2
    data.loc[data['famer_product_name']== 'leicaxvarioapscdigitalcamera18430', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'panasonicdmcfz70', 'famer_digital_zoom'] = 5
    data.loc[data['famer_product_name']== 'ricohgrwideanglelcd175743', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'ricohgrlimitededwhoodstrapcase175823', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'nikoncoolpixp7800', 'famer_digital_zoom'] = 4
    data.loc[data['famer_product_name']== 'fujifilmx100t', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'leicax2', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'sigmadp1', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'fujifilmx100t', 'famer_digital_zoom'] = np.nan
    data.loc[data['famer_product_name']== 'sigmadp2', 'famer_digital_zoom'] = np.nan

    data.to_csv(path_to_cleaned_data + 'henrys_data_cleaned.csv')





def clean_ilgs_data(data):
    # Optical zoom
    data['famer_opticalzoom'] = np.where(data['famer_product_name']== 'fujifilms9200', 50, data['famer_opticalzoom'])
    data['famer_opticalzoom'] = np.where(data['famer_product_name']== 'sonyqx10', 10, data['famer_opticalzoom'])
    data['famer_opticalzoom'] = np.where(data['famer_product_name']== 'canondigitalixus140', 8, data['famer_opticalzoom'])
    data['famer_opticalzoom'] = np.where(data['famer_product_name']== 'sonywx300', 20, data['famer_opticalzoom'])
    data['famer_opticalzoom'] = np.where(data['famer_product_name']== 'sonyhx300', 50, data['famer_opticalzoom'])
    # digital zoom
    data['extracted_digital_zoom'] = data['digital zoom'].str.extract(r'([\d.]+)\s*x')
    # sensor typ
    data['extracted_sensor_typ'] = data['sensor type'].apply(assign_sensor_type)
    # weight
    #ilgs_data_non_null_columns['extracted_weight'] = ilgs_data_non_null_columns['weight including battery']
    data['extracted_weight'] = data['weight including battery'].fillna(data['weight'])
    data['extracted_weight'] = data['extracted_weight'].str.extract(r'([\d.]+)\s*g')
    # width
    data['extracted_width'] = data['dimensions w x d x h'].apply(lambda x: extract_dimension(x,'width',0))
    data['extracted_width'] = data['extracted_width'].str.replace('mm','')
    data['extracted_width'] = data['extracted_width'].apply(convert_mm_to_cm)
    data['extracted_width'] = data['extracted_width'].fillna(data['famer_width'])

    # height
    data['extracted_height'] = data['dimensions w x d x h'].apply(lambda x: extract_dimension(x,'height',2))
    data['extracted_height'] = data['extracted_height'].str.replace('mm','')
    data['extracted_height'] = data['extracted_height'].apply(convert_mm_to_cm)


    # Columns to be removed
    #columns_to_be_removed = ['famer_width', 'famer_sensortype', 'dimensions w x d x h','display resolution numeric',
    #                         'sensor type','optical zoom','weight','famer_weight','digital zoom', 'weight including battery']

    # Remove the specified columns
    #data.drop(columns= columns_to_be_removed, axis=1, inplace=True)


    ilgs_data_new_column_names = {
        'extracted_digital_zoom': 'famer_digital_zoom',
        'famer_opticalzoom': 'famer_optical_zoom',
        'extracted_width': 'famer_width',
        'extracted_height': 'famer_height',
        'extracted_weight': 'famer_weight',
        'extracted_sensor_typ': 'famer_sensor'
    }

    data.rename(columns=ilgs_data_new_column_names, inplace=True)


    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'ilgs_data_cleaned.csv')




def clean_canon_europe_data(data):
    # Optical and digital Zoom
    data['optical_zoom'] = data['zoom'].str.extract(r'Optical ([\d.]+)x')
    data['digital_zoom'] = data['zoom'].str.extract(r'Digital Approx.? ([\d.]+)x',flags=re.I)

    # Weight
    data['extracted_weight'] = data['weight']
    data['extracted_weight'] = data['extracted_weight'].fillna(data['weight body only'])
    data['extracted_weight'] = data['extracted_weight'].str.extract(r'([\d.]+)\s*g')
    # dimesnions
    data['dimensions'] = data['dimensions'].fillna(data['dimensions wxhxd'])

    # width
    data['width'] = data['dimensions'].apply(lambda x: extract_dimension(x,'width',0))
    data['width'] = data['width'].str.replace('mm','')
    data['width'] = data['width'].apply(convert_mm_to_cm)

    # height
    data['height'] = data['dimensions'].apply(lambda x: extract_dimension(x,'height',1))
    data['height'] = data['height'].str.replace('mm','')
    data['height'] = data['height'].apply(convert_mm_to_cm)


    # Columns to be removed
    #columns_to_be_removed = ['zoom','weight','dimensions wxhxd','dimensions', 'weight body only']

    # Remove the specified columns
    #canon_europe_data_non_null_columns.drop(columns= columns_to_be_removed, axis=1, inplace=True)


    canon_data_new_column_names = {
        'optical_zoom': 'famer_optical_zoom',
        'digital_zoom': 'famer_digital_zoom',
        'extracted_weight': 'famer_weight',
        'width': 'famer_width',
        'height': 'famer_height'
     }

    data.rename(columns=canon_data_new_column_names, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)
    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'canon_europe_data_cleaned.csv')




def clean_camerafarm_data(data):
    # Sensor type
    data['sensor type'] = data['sensor type'].fillna(data['image sensor type'])

    # weight
    data['weight'] = data['weight'].fillna(data['approx weight'])
    data['extracted_weight'] =  data['weight'].str.extract(r'([\d.]+)\s*g')
    data['extracted_weight'] =  data['extracted_weight'].fillna(data['famer_weight'])

    # width
    data['width_approx_dim'] = data['approx dimensions'].str.extract(r'Width\s*\\?:?\s*([\d.]+)')
    data['width_approx_dim'] = data['width_approx_dim'].apply(lambda x: convert_dimension_to_cm(x,'inch'))
    data['width'] = data['dimensions wxhxd'].apply(lambda x: extract_dimension(x,'width',0))
    pattern = r'[mmW\(\)]'
    data['width'] = data['width'].str.replace(pattern, '')
    data['width'] = data['width'].apply(convert_mm_to_cm)
    data['width'] = data['width'].fillna(data['width_approx_dim'])

    # height
    data['height_approx_dim'] = data['approx dimensions'].str.extract(r'Height\s*\\?:?\s*([\d.]+)')
    data['height_approx_dim'] = data['height_approx_dim'].apply(lambda x: convert_dimension_to_cm(x,'inch'))
    data['height'] = data['dimensions wxhxd'].apply(lambda x: extract_dimension(x,'height',1))
    pattern = r'[mmH\(\)]'
    data['height'] = data['height'].str.replace(pattern, '')
    data['height'] = data['height'].apply(convert_mm_to_cm)
    data['height'] = data['height'].fillna(data['height_approx_dim'])

    # Columns to be removed
    #columns_to_be_removed = ['image sensor type','dimensions wxhxd','weight', 'approx weight',
    #                        'approx dimensions','resolution','width_approx_dim','height_approx_dim']

    # Remove the specified columns
    #data.drop(columns= columns_to_be_removed, axis=1, inplace=True)


    camera_farm_data_new_column_names = {
        'extracted_weight': 'famer_weight_real',
        'sensor type': 'famer_sensor',
        'width': 'famer_width',
        'height': 'famer_height'
    }

    data.rename(columns=camera_farm_data_new_column_names, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)
    data['famer_weight'] = data['famer_weight_real']
    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'camera_farm_data_new.csv')






def clean_ebay_data(data):
    # read a dataframe that contains manually cleaned infos about the weights
    cleaned_dimensions_dataframe = pd.read_csv(path_to_help_data + 'dimensions_thrid_columns.csv', sep=",")

    # Merge dataframes on the common column
    data = pd.merge(data, cleaned_dimensions_dataframe[['extracted_width_cm', 'extracted_height_cm','famer_product_name']], on='famer_product_name', how='left')


    # Extract the Weight values
    data['extracted_weight'] = data.apply(consolidate_weight_columns, axis=1)
    # Apply the function to 'temp_extracted_weight' column and create a new column 'extracted_weight'
    data['extracted_weight_g'] = data['extracted_weight'].str.extract(r'([\d.]+)\s*g',flags=re.IGNORECASE)
    data['extracted_weight_Oz'] = data['extracted_weight'].str.extract(r'([\d.]+)\s*Oz',flags=re.IGNORECASE) + " Oz"
    data['extracted_weight_Kg'] = data['extracted_weight'].str.extract(r'([\d.]+)\s*Kg',flags=re.IGNORECASE) + " Kg"
    # convert oz into g
    data['extracted_weight_Oz'] = data['extracted_weight_Oz'].apply(convert_weight_to_grams)
    # convert kg into g
    data['extracted_weight_Kg'] = data['extracted_weight_Kg'].apply(convert_weight_to_grams)
    # fill nan values (needs to check the famer_weight)
    data['extracted_weight_g'] = data['extracted_weight_g'].fillna(data['extracted_weight_Oz']).fillna(data['extracted_weight_Kg'])


    data.drop('famer_weight', axis=1, inplace=True)
    # Extract the sensor type
    data['sensor type'] = data['sensor type'].str.split('\\', expand=True)[0]

    ebay_data_new_column_names = {
            'extracted_weight_g': 'famer_weight',
            'sensor type': 'famer_sensor',
            'extracted_width_cm': 'famer_width',
            'extracted_height_cm': 'famer_height',
            'famer_opticalzoom': 'famer_optical_zoom'
        }

    data.rename(columns=ebay_data_new_column_names, inplace=True)


    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]
    data.to_csv(path_to_cleaned_data + 'ebay_data_cleaned.csv')





def clean_mypriceindia_data(data):
    # Weight in g
    # Width in cm
    # Height in cm

    # optical zoom
    data['optical zoom'] = data['optical zoom'].str.extract(r'([\d.]+)\s*x')

    # Digital zoom
    data['digital zoom'] = data['digital zoom'].str.extract(r'([\d.]+)\s*x')


    # Width
    data['extracted_width'] = data['dimensions'].apply(lambda x: extract_dimension(x,'width',0))
    data['extracted_width'] = data['extracted_width'].str.replace(r'[^\d.]+', '', regex=True)
    data['extracted_width'] = pd.to_numeric(data['extracted_width'], errors='coerce')
    data['extracted_width'] = data['extracted_width'] / 10.0


    # Height
    data['extracted_height'] = data['dimensions'].apply(lambda x: extract_dimension(x,'height',1))
    data['extracted_height'] = data['extracted_height'].str.replace(r'[^\d.]+', '', regex=True)
    data['extracted_height'] = pd.to_numeric(data['extracted_height'], errors='coerce')
    data['extracted_height'] = data['extracted_height'] / 10.0

    # Weight
    data['extracted_weight'] = data['weight'].str.extract(r'(?i)(\d+)\s*g \(with Batt')
    data['extracted_weight'] = data['extracted_weight'].fillna(data['famer_weight'])

    # Columns to be removed
    columns_to_be_removed = ['weight','famer_sensortype','dimensions','video resolution','famer_weight']

    # Remove the specified columns
    data.drop(columns= columns_to_be_removed, axis=1, inplace=True)

    mypriceindia_data_new_column_names = {
        'extracted_weight': 'famer_weight',
        'sensor type': 'famer_sensor',
        'digital zoom': 'famer_digital_zoom',
        'optical zoom': 'famer_optical_zoom',
        'extracted_width':'famer_width',
        'extracted_height': 'famer_height'
    }

    columns_to_order = ['key', 'source','famer_model_list','famer_model_no_list','famer_mpn_list','famer_ean_list',
                        'famer_product_name', 'famer_digital_zoom','famer_optical_zoom', 'famer_width', 'famer_height',
                        'famer_weight', 'famer_sensor','famer_brand_list','famer_resolution_from','famer_resolution_to','famer_megapixel']

    data.rename(columns=mypriceindia_data_new_column_names, inplace=True)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]

    data.to_csv(path_to_cleaned_data + 'mypriceindia_data_cleaned.csv')



def clean_shopmania_data(data):
    # Weight in g
    # Width in cm
    # height in cm

    # Extract weight in g
    data['extracted_weight'] = data['weight'].str.replace(r'\D', '', regex=True)

    # Extract digital zoom
    data['extracted_digital_zoom'] = data['digital zoom'].str.replace(r'\D', '', regex=True)

    # Extract width in cm
    data['extracted_width'] = pd.to_numeric(data['famer_width'], errors='coerce')
    data['extracted_width'] = data['extracted_width'] / 10.0


    # Extract height in cm
    data['extracted_height'] = data['height'].str.replace(r'[^\d.]+', '', regex=True)
    data['extracted_height'] = pd.to_numeric(data['extracted_height'], errors='coerce')
    data['extracted_height'] = data['extracted_height'] / 10.0

    columns_to_drop = ['famer_width']
    data = data.drop(columns = columns_to_drop)

    mypriceindia_data_new_column_names = {
        'sensor type': 'famer_sensor',
        'extracted_weight': 'famer_weight',
        'extracted_digital_zoom': 'famer_digital_zoom',
        'extracted_width':'famer_width',
        'extracted_height': 'famer_height',
        'famer_opticalzoom': 'famer_optical_zoom'
    }

    data = data.rename(columns = mypriceindia_data_new_column_names)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]

    data.to_csv(path_to_cleaned_data + 'shopmania_data_cleaned.csv')


def clean_ukdigitalcameras_data(data):

    data.rename(columns={'famer_opticalzoom': 'famer_optical_zoom'}, inplace=True)

    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]

    data.to_csv(path_to_cleaned_data + 'ukdigitalcameras_data_cleaned.csv')


def clean_walmart_data(data):
    # Extract Width
    data['famer_width'] = data['product in inches l x w x h'].str.split(' x ').str[1].astype(float)
    data['famer_width'] = data['famer_width'] * 2.54


    # Extract Height
    data['famer_height'] = data['product in inches l x w x h'].str.split(' x ').str[2].astype(float)
    data['famer_height'] = data['famer_height'] * 2.54



    # Extract Weight
    data['famer_weight'] = data['famer_weight']
    data['famer_weight'] = data['famer_weight'] * 453.592


    # Extract optical zoom
    data.rename(columns={'famer_opticalzoom': 'famer_optical_zoom'}, inplace=True)


    # Extract digital zoom
    data['digital zoom'] = data['digital zoom'].str.extract('(\d+)')
    data.rename(columns={'digital zoom': 'famer_digital_zoom'}, inplace=True)
    data = ensure_columns_exist(data, columns_to_order)
    data = data[columns_to_order]

    data.to_csv(path_to_cleaned_data + 'walmart_data_cleaned.csv')



def clean_pricedekho_data(data):
    """
    Cleans the Pricedekho camera dataset by extracting and converting key features such as weight, dimensions, and zoom values.

    Parameters:
    data (pd.DataFrame): The raw dataset containing camera information.

    Returns:
    pd.DataFrame: The cleaned dataset with extracted and converted features.
    """

    # Define subfunctions for extracting weight, optical zoom, and digital zoom
    def pricedekho_extract_weight(text):
        if not pd.isna(text):
            match = re.search(r'Weight\s*\\n\s*(\d+)', text)
            if match:
                return match.group(1)
            else:
                return None

    def pricedekho_extract_optical_zoom(text):
        if not pd.isna(text):
            match = re.search(r'Optical Zoom\s*\\n\s*(\d+)', text)
            if match:
                return match.group(1)
            else:
                return None

    def pricedekho_extract_digital_zoom(text):
        if not pd.isna(text):
            match = re.search(r'Digital Zoom\s*\\n\s*(\d+)', text)
            if match:
                return match.group(1)
            else:
                return None

    # Extract columns with at least one non-null value
    data = data.dropna(axis=1, how='all')

    # Specify the index of the column to rename
    column_index = 110
    # Get the current column names
    current_columns = list(data.columns)

    # Modify the name of the column at the specified index
    new_column_name = 'sensor_typ_2'
    current_columns[column_index] = new_column_name

    # Assign the modified list of column names back to the DataFrame
    data.columns = current_columns

    # Define the important features to keep
    pricedekho_cameras_data_important_features = ['key', 'source', 'famer_brand_list', 'famer_model_list',
                                                  'famer_product_name', 'digital zoom', 'famer_opticalzoom',
                                                  'optical zoom', 'sensor type', 'sensor_typ_2',
                                                  'weight', 'dimension', 'dimensions', 'display', 'zoom',
                                                  'famer_resolution_from', 'famer_resolution_to']
    data = data[pricedekho_cameras_data_important_features]

    # Extract and convert dimensions
    data['extracted_width'] = data['dimensions'].apply(lambda x: extract_dimension(x, 'width', 0)).apply(convert_dimension_to_cm)
    data['extracted_height'] = data['dimensions'].apply(lambda x: extract_dimension(x, 'height', 1)).apply(convert_dimension_to_cm)

    # Extract and clean weight data
    data['extracted_weight'] = data['weight'].str.extract('(\d+)')
    data['extracted_weight_second'] = data['dimension'].apply(pricedekho_extract_weight)
    data['extracted_weight_final'] = data['extracted_weight'].fillna(data['extracted_weight_second'])

    # Extract and clean optical zoom data
    data['optical zoom'] = data['optical zoom'].replace('[^0-9.]', '', regex=True)
    data['new_optical_zoom'] = data['zoom'].apply(pricedekho_extract_optical_zoom)
    data['final_extracted_optical_zoom'] = data['optical zoom'].fillna(data['new_optical_zoom'])
    data.loc[data['famer_product_name'] == 'canoneos550d', 'final_extracted_optical_zoom'] = 15

    # Extract and clean digital zoom data
    data['digital zoom'] = data['digital zoom'].replace('[^0-9.]', '', regex=True)
    data['new_digital_zoom'] = data['zoom'].apply(pricedekho_extract_digital_zoom)
    data['final_extracted_digital_zoom'] = data['digital zoom'].fillna(data['new_digital_zoom'])
    data.loc[data['famer_product_name'] == 'sonycybershotdscw730', 'final_extracted_digital_zoom'] = 32
    data.loc[data['famer_product_name'] == 'samsungsh100', 'final_extracted_digital_zoom'] = 5
    data.loc[data['famer_product_name'] == 'fujifilmfinepixs6800', 'final_extracted_digital_zoom'] = 2
    data.loc[data['famer_product_name'] == 'samsungwb100', 'final_extracted_digital_zoom'] = 2
    data.loc[data['famer_product_name'] == 'panasoniclumixdmcgx1w', 'final_extracted_digital_zoom'] = 4
    data.loc[data['famer_product_name'] == 'samsungpl120', 'final_extracted_digital_zoom'] = 5

    # Clean and finalize sensor type data
    data['sensor type'] = data['sensor type'].str.replace('Sensor', '')
    data['sensor_typ_2'] = data['sensor_typ_2'].str.replace('Sensor', '')
    data['final_extracted_sensor_typ'] = data['sensor type'].fillna(data['sensor_typ_2'])

    # Rename columns to standardized names
    pricedekho_data_new_column_names = {
        'final_extracted_sensor_typ': 'famer_sensor',
        'extracted_weight_final': 'famer_weight',
        'final_extracted_digital_zoom': 'famer_digital_zoom',
        'extracted_width': 'famer_width',
        'extracted_height': 'famer_height',
        'final_extracted_optical_zoom': 'famer_optical_zoom'
    }
    data = data.rename(columns=pricedekho_data_new_column_names)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    # Reorder columns to a predefined order
    data = data[columns_to_order]

    # Save the cleaned data to a CSV file
    data.to_csv(path_to_cleaned_data + 'pricedekho_data_cleaned.csv', index=False)

    return data




def clean_cambuy_data(data):

    # Weight in g
    # Width in cm
    # Height in cm

    # Extract digital zoom
    data['digital zoom'] = data['digital zoom'].replace('[^0-9.]', '', regex=True)
    data.loc[data['famer_product_name'] == 'panasoniclumixdmctz60', 'digital zoom'] = 4
    data.loc[data['famer_product_name'] == 'ricohwg4', 'digital zoom'] = 7.2
    data.loc[data['famer_product_name'] == 'panasoniclumixdmcgx7', 'digital zoom'] = 2
    data.loc[data['famer_product_name'] == 'panasoniclumixdmclx100', 'digital zoom'] = 4
    data.loc[data['famer_product_name'] == 'olympusstylus1', 'digital zoom'] = 4


    # Extract Weight
    data['extracted_weight'] = data['weight'].str.extract(r'Approx\. (\d+)')
    data.loc[data['famer_product_name']== 'panasoniclumixdmctz60', 'weight'] = 240
    data.loc[data['famer_product_name']== 'panasoniclumixdmclx100', 'weight'] = 393
    data.loc[data['famer_product_name']=='panasoniclumixdmcfz1000', 'weight'] = 831

    # Extract Width
    data['dimensions w x h x d'] = data['dimensions w x h x d'].str.replace('Approx. ', '')
    data['extracted_width'] = data['dimensions w x h x d'].apply(lambda x: extract_dimension(x,'width',0))
    data.loc[data['famer_product_name']== 'nikondfdslr', 'extracted_width'] = 143.5
    data.loc[data['famer_product_name']== 'nikondfdslr', 'extracted_width'] = 143.5
    data.loc[data['famer_product_name']== 'nikond750', 'extracted_width'] = 140.5
    data.loc[data['famer_product_name']== 'nikond810', 'extracted_width'] = 146

    data['extracted_width'] = data['extracted_width'].replace('[^0-9.]', '', regex=True)
    data['extracted_width'] = pd.to_numeric(data['extracted_width'])
    data.loc[data['extracted_width'].notnull(), 'extracted_width'] /= 10

    # Extract Height
    data['extracted_height'] = data['dimensions w x h x d'].apply(lambda x: extract_dimension(x,'height',1))
    data.loc[data['famer_product_name']== 'nikondfdslr', 'extracted_height'] = 110
    data.loc[data['famer_product_name']== 'nikond810', 'extracted_height'] = 123
    data.loc[data['famer_product_name']== 'nikond750', 'extracted_height'] = 113

    data['extracted_height'] = data['extracted_height'].replace('[^0-9.]', '', regex=True)
    data['extracted_height'] = pd.to_numeric(data['extracted_height'])
    data.loc[data['extracted_height'].notnull(), 'extracted_height'] /= 10


    cambuy_cameras_data_new_column_names = {
        'digital zoom': 'famer_digital_zoom',
        'famer_opticalzoom': 'famer_optical_zoom',
        'extracted_weight': 'famer_weight',
        'extracted_width':'famer_width',
        'extracted_height': 'famer_height'
    }

    data = data.rename(columns=cambuy_cameras_data_new_column_names)

    # Check and create missing columns
    data = ensure_columns_exist(data, columns_to_order)

    data = data[columns_to_order]

    data.to_csv(path_to_cleaned_data + 'cambuy_data_cleaned.csv')




def clean_data(data,to_duplicate):

    # Define important features for each source in a dictionary
    source_features = {
        'www.priceme.co.nz': ['key', 'source','famer_brand_list','famer_keys','famer_model_list','famer_model_no_list','famer_product_name','recId'],
        'buy.net':           ['key', 'source','famer_keys' ,'famer_brand_list', 'famer_model_list','famer_model_no_list','famer_keys', 'famer_product_name',
                              'weight', 'digital zoom', 'optical zoom', 'height', 'width', 'image sensor','recId'],
        'www.buzzillions.com' : ['key','key', 'source','famer_brand_list', 'famer_model_list','famer_model_no_list','famer_keys','famer_product_name',
                                 'famer_weight', 'optical zoom', 'digital zoom','height', 'width','famer_megapixel',
                                 'sensor','famer_mpn_list','famer_ean_list','recId'],
        'www.eglobalcentral.co.uk' : ['key','source','famer_brand_list', 'famer_model_list','famer_model_no_list','famer_keys','famer_product_name',
                                      'digital zoom', 'optical zoom', 'dimensions w x h x d','dimensions', 'famer_sensortype','famer_resolution_from','famer_resolution_to','recId'],
        'www.price-hunt.com':        ['key','source','famer_brand_list', 'famer_model_list','famer_model_no_list','famer_product_name',
                                      'digital zoom','optical zoom','weight', 'sensor type', 'dimensions','famer_resolution_from','famer_resolution_to',
                                      'optical sensor resolution in megapixel','other resolution','weight without battery','recId'],
        'www.shopbot.com.au':        ['key' ,'source','famer_brand_list', 'famer_keys','famer_model_list','famer_model_no_list','famer_product_name',
                                      'zoom', 'weight', 'dimensions','video resolution','recId'],
        'www.wexphotographic.com':   ['key', 'famer_keys','source', 'famer_brand_list', 'famer_keys','famer_model_list',
                                      'famer_model_no_list','famer_product_name', 'optical zoom x', 'weight g',
                                      'sensor type', 'resolution', 'size','recId'],
        'www.garricks.com.au':       ['key', 'famer_keys','source', 'famer_brand_list','famer_keys','famer_model_list','famer_model_no_list'
                                     ,'famer_product_name','zoom range', 'weight','sensor details',
                                     'resolution','recId'],
        'www.pcconnection.com':      ['key', 'source','famer_brand_list', 'famer_keys','famer_model_no_list','famer_model_list', 'famer_product_name',
                                        'digital zoom', 'optical zoom','actual weight', 'optical sensor type',
                                        'physical dimensions', 'optical sensor type','recId'],
        'cammarkt.com':             ['key', 'source','famer_keys', 'famer_brand_list', 'famer_keys','famer_model_list','famer_model_no_list','famer_product_name',
                                        'image sensor type','famer_opticalzoom', 'optical zoom','digital zoom','weight',
                                        'famer_weight', 'image resolutions','megapixels','famer_ean_list','height', 'width',
                                        'famer_width','camera resolution','famer_height','recId'],
        'www.gosale.com':           ['key','famer_keys' ,'source', 'famer_brand_list', 'famer_keys',
                                      'famer_model_list','famer_model_no_list','famer_product_name','famer_weight',
                                      'famer_opticalzoom','famer_ean_list','famer_mpn_list','product number mpn','recId'
                                    ],
        'www.henrys.com':           ['key','famer_keys' ,'source','famer_brand_list', 'famer_keys','famer_model_list','famer_model_no_list',
                                      'famer_product_name','digital zoom', 'optical zoom','weight', 'dimensions wxhxd', 'famer_opticalzoom',
                                      'sensor typesize','video resolution', 'famer_weight','recId'],
        'www.ilgs.net':             ['key','famer_keys' ,'source', 'famer_brand_list','famer_keys','famer_model_list','famer_model_no_list','famer_product_name','famer_opticalzoom','famer_width',
                                     'famer_sensortype',
                                     'digital zoom', 'optical zoom','weight','famer_weight','weight including battery', 'dimensions w x d x h',
                                     'display resolution numeric', 'famer_ean_list','famer_mpn_list',
                                     'sensor type','recId'],
        'www.canon-europe.com':      ['key', 'source', 'famer_keys','famer_brand_list','famer_keys','famer_model_list',
                                        'famer_model_no_list','famer_product_name','zoom','weight',
                                        'dimensions wxhxd','dimensions','weight body only','recId'],
        'www.camerafarm.com.au':      ['key', 'source', 'famer_brand_list', 'famer_model_list','famer_model_no_list','famer_keys',
                                        'famer_product_name', 'image sensor type','sensor type','famer_weight',
                                        'weight','approx weight','dimensions wxhxd', 'approx dimensions',
                                        'resolution','recId'],
        'www.ebay.com'          :       ['key', 'source', 'famer_brand_list','famer_model_list','famer_model_no_list', 'famer_product_name','famer_ean_list',
                                          'famer_mpn_list','famer_weight', 'weight', 'approx weight', 'camera weight', 'sensor type','famer_opticalzoom','recId',
                                          'famer_resolution_from','famer_resolution_to','famer_megapixel'
                                        ],

        'www.mypriceindia.com'  :       ['key', 'source', 'famer_brand_list','famer_model_list', 'famer_product_name', 'weight', 'famer_model_no_list',
                                         'famer_weight', 'digital zoom', 'optical zoom', 'famer_sensortype',
                                         'sensor type','dimensions','video resolution','famer_resolution_from','famer_resolution_to'],

        'www.shopmania.in'      :       ['key', 'source', 'famer_brand_list', 'famer_model_list',
                                         'famer_product_name', 'weight','famer_opticalzoom','digital zoom',
                                         'optical zoom', 'sensor type','resolution', 'image resolutions','ean',
                                         'famer_ean_list','famer_width','width', 'height','famer_resolution_from','famer_resolution_to'
                                        ],

        'www.ukdigitalcameras.co.uk':   ['key','source', 'famer_brand_list', 'famer_model_list', 'famer_product_name', 'famer_opticalzoom' ,
                                         'optical zoom','camera resolution', 'famer_mpn_list'],

        'www.walmart.com':              ['key', 'source','famer_brand_list','famer_model_list',
                                         'famer_product_name', 'digital zoom' ,'famer_opticalzoom' ,
                                         'optical zoom', 'famer_weight','product in inches l x w x h',
                                         'resolution megapixels','famer_megapixel'],

        'www.cambuy.com.au':            ['key', 'source', 'famer_brand_list', 'famer_model_list',
                                         'famer_product_name', 'digital zoom','famer_opticalzoom',
                                         'optical zoom','weight', 'image sensor','dimensions w x h x d']



    }

    # Dictionary to store cleaning functions for each source
    cleaning_functions = {
        'www.priceme.co.nz': clean_priceme_data,
        'buy.net': clean_buy_data,
        'www.buzzillions.com': clean_buzzillions_data,
        'www.eglobalcentral.co.uk': clean_eglobalcentral_data,
        'www.price-hunt.com': clean_price_hunt_data,
        'www.shopbot.com.au': clean_shopbot_data,
        'www.wexphotographic.com': clean_wexphotographic_data,
        'www.garricks.com.au':   clean_garricks_data,
        'www.pcconnection.com': clean_pcconnection_data,
        'cammarkt.com': clean_cammarkt_data,
        'www.gosale.com': clean_gosale_data,
        'www.henrys.com': clean_henrys_data,
        'www.ilgs.net': clean_ilgs_data,
        'www.canon-europe.com': clean_canon_europe_data,
        'www.camerafarm.com.au': clean_camerafarm_data,
        'www.ebay.com': clean_ebay_data,
        'www.mypriceindia.com':clean_mypriceindia_data,
        'www.shopmania.in': clean_shopmania_data,
        'www.ukdigitalcameras.co.uk':clean_ukdigitalcameras_data,
        'www.walmart.com':clean_walmart_data,
        'www.pricedekho.com': clean_pricedekho_data,
        'www.cambuy.com.au':clean_cambuy_data
    }


    # Clean data based on source
    for source, cleaning_function in cleaning_functions.items():
        print("The source we are now processing is ")
        print(source)
        data_temp = data.loc[data['source'] == source]
        if not to_duplicate:
            # and source not in ['www.ebay.com']:
            # Remove duplicates from the data sources
            data_temp = data_temp.drop_duplicates(subset='recId')
        print("It has {} rows and {} columns".format(data_temp.shape[0],data_temp.shape[1]))
        if source != 'www.pricedekho.com':
            data_important_features = source_features[source]
            data_cleaned = data_temp[data_important_features].dropna(how='all')
            cleaning_function(data_cleaned)
        else:
            cleaning_function(data_temp)



def prepare_and_clean_data(to_duplicate=False):

    # Read data
    data = pd.read_csv(path_to_raw_data + 'camera_raw_data.csv')

    # Clean data
    clean_data(data,to_duplicate)


prepare_and_clean_data(to_duplicate=False)




