"""
My draft for reading in the code contained in the csv files.

Medical_Keyword_1-48 are dummy variables.
"""
import pandas as pd
import numpy as np

def get_data():
    # Hardcoding in the paths here
    TRAIN_PATH = "./train.csv"
    TEST_PATH = "./test.csv"
    
    # Import via pandas
    old_train = pd.read_csv(TRAIN_PATH)
    old_test = pd.read_csv(TEST_PATH)
    converted_train_list = []
    converted_test_list = []  # I will later use pandas to get a full df
    
    # Make lists for conversions
    categorical_data_list = [ "Product_Info_1", "Product_Info_2", 
                              "Product_Info_3", "Product_Info_5", 
                              "Product_Info_6", "Product_Info_7", 
                              "Employment_Info_2", "Employment_Info_3", 
                              "Employment_Info_5", "InsuredInfo_1", 
                              "InsuredInfo_2", "InsuredInfo_3", 
                              "InsuredInfo_4", "InsuredInfo_5", 
                              "InsuredInfo_6", "InsuredInfo_7", "Insurance_History_1", 
                              "Insurance_History_2", "Insurance_History_3", 
                              "Insurance_History_4", "Insurance_History_7", 
                              "Insurance_History_8", "Insurance_History_9", 
                              "Family_Hist_1", "Medical_History_2", 
                              "Medical_History_3", "Medical_History_4", 
                              "Medical_History_5", "Medical_History_6", 
                              "Medical_History_7", "Medical_History_8", 
                              "Medical_History_9", "Medical_History_11", 
                              "Medical_History_12", "Medical_History_13", 
                              "Medical_History_14", "Medical_History_16", 
                              "Medical_History_17", "Medical_History_18", 
                              "Medical_History_19", "Medical_History_20", 
                              "Medical_History_21", "Medical_History_22", 
                              "Medical_History_23", "Medical_History_25", 
                              "Medical_History_26", "Medical_History_27", 
                              "Medical_History_28", "Medical_History_29", 
                              "Medical_History_30", "Medical_History_31", 
                              "Medical_History_33", "Medical_History_34", 
                              "Medical_History_35", "Medical_History_36", 
                              "Medical_History_37", "Medical_History_38", 
                              "Medical_History_39", "Medical_History_40", 
                              "Medical_History_41" ]
    
    continuous_data_list = [ "Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", 
                             "Employment_Info_1", "Employment_Info_4", 
                             "Employment_Info_6", "Insurance_History_5", 
                             "Family_Hist_2", "Family_Hist_3", 
                             "Family_Hist_4", "Family_Hist_5" ]
    
    discrete_data_list = [ "Medical_History_1", "Medical_History_10", 
                           "Medical_History_15", "Medical_History_24", 
                           "Medical_History_32" ]
    
    # Convert categorical data use pandas get_dummies
    for category in categorical_data_list:
        # First for training
        dummies = pd.get_dummies(old_train[category], dummy_na=False)
        converted_train_list.append(dummies)
        
        # Then for testing
        dummies = pd.get_dummies(old_test[category], dummy_na=False)
        converted_test_list.append(dummies)
        
    # Convert continuous data to float32
    df = old_train[continuous_data_list].convert_objects(convert_numeric=True)
    tf = old_test[continuous_data_list].convert_objects(convert_numeric=True)
    # I don't know how appending a list of dataframes will work, should be fine
    converted_train_list.append(df); converted_test_list.append(tf) 
        
    # Convert Discrete data to variables (don't know how it really looks atm)
    for category in discrete_data_list:
        # First for training
        dummies = pd.get_dummies(old_train[category], dummy_na=False)
        converted_train_list.append(dummies)
        
        # Then for testing
        dummies = pd.get_dummies(old_test[category], dummy_na=False)
        converted_test_list.append(dummies)
    
    # Make the full dataframes here
    train = pd.concat(converted_train_list,axis=1)
    test = pd.concat(converted_test_list,axis=1)
    
    # So far I've made the assumption that there are no new variables or
    # features in the test dataset vs. the train dataset. This will rectify that
    columns_to_keep = list(train.columns.values)
    
    """ Prints for Debugging """
    #print list(train.columns.values)
    #print list(test.columns.values)
    print train.columns
    print test.columns

    # Get the y_data bits
    y_train = old_train["Response"]
    test_id = old_test["Id"]

    # To Return
    x_train = train[columns_to_keep];
    x_test = test[columns_to_keep];    
    # Returning an error: 
    # IndexError: index 4540 is out of bounds for axis 1 with size 1679

    # Return everything
    return x_train, y_train, x_test, test_id;
