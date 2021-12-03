"""
Functions to determine feature importance from models. Returns a dataframe with r2 or AUC_ROC scores. The score that is the lowest has the most importance.
"""

import tempfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
import random
import json


from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.utils.struct_utils import base_smiles_from_smiles
from atomsci.ddm.pipeline import predict_from_model as pfm

#***************************************
def _making_dataframe(model_type, pred_df,auc_roc_list_grid,r2_list_grid,response_col):
    """
    A helper function assembles and returns the final dataframe.
    """
    if model_type == 'regression':
        r2_scored = r2_score(y_true = pred_df[f'{response_col}_actual'], y_pred = pred_df[f'{response_col}_pred'])
        r2_list_grid.append(r2_scored)
    elif model_type == 'classification':
        auc_roc_scored = roc_auc_score(y_true = pred_df[f'{response_col}_actual'], y_score = pred_df[f'{response_col}_pred'])
        auc_roc_list_grid.append(auc_roc_scored)
#***************************************

    
    
#***************************************
def _shuffle_columns(tempFeaturizer,j, columns, feature_list_grid):
    """
    A helper function that does the shuffling of the columns.
    """
    temp_feature = columns[-j]
    print(temp_feature)
    temp_col = tempFeaturizer[temp_feature]
    temp_col = temp_col.tolist()
    random.shuffle(temp_col)
    tempFeaturizer[temp_feature] = temp_col
    feature_list_grid.append(temp_feature)
    return tempFeaturizer
#***************************************
        
    

def predict_feature_importance(model_type, model_path, input_df, id_col, smiles_col, response_col,featurizer_kind = None,n_cols = None):
    """
    Loads a pretrained model from the model tarball file and runs predictions on feature importance by shuffling the feature  columns for given descriptors
    
    Args:
        model_type (str): Input of the model type can be Reg (regression) or Class (classification).
        
        featurizer_kind (str): Input of the type of featurizer that can be used (moe, moe_raw, moe_scaled, moe_norm, moe_filtered, moe_scaled_filtered, moe_informative, moe_scaled, mordred_filtered, mordred_raw, rdkit_raw)
        
 
        
        model_path: The path to take to reach the tarball file
        
        input_df (DataFrame): The input DataFrame that contains id_col, smiles_col, response_col
        
        id_col (str): Name of the column containing compound IDs. If none is provided, sequential IDs will be
        generated.
        
        smiles_col (str): Name of the column containing SMILES strings; required.
        
        response_col (str): Name of an optional column containing actual response values; if it is provided,
        the actual values will be included in the returned data frame to make it easier for you to assess performance.
        
        n_cols (int): This is set to none as default. If a user would like to use their own data set, they may enter in the length of their dataset. It must be noted that the feature column must be last.
        
        
    Return:
        A data frame with the name of the feature column that was shuffled and a column with the corresponding AUC_ROC or r2 score. Note, all parameters must be present for the function to work.
    """
    #Infer the featurizer kind, model type from the tarball
    #Set featurizer kind automatically
    #Check input dataframe
    #Combine the two input definitions (Pred_df and data_to_predict)
    
    #***************************************
    tempFeaturizer = input_df.copy()
    columns = list(tempFeaturizer.columns)
    
    feature_list_grid = []
    r2_list_grid = []
    auc_roc_list_grid = []
    #***************************************
    
    #***************************************
    #finding the featurizing kind here
    #file_path =
    #file = open
    #featurizer_kind =
    
    
    #***************************************
    
    
    
    
    #***************************************
    if featurizer_kind == 'moe'or featurizer_kind == 'moe_norm'or featurizer_kind == 'moe_filtered' or featurizer_kind == 'moe_scaled_filtered':
        numberOfColumns = 306
    elif featurizer_kind == 'moe_raw':
        numberOfColumns = 332
    elif featurizer_kind == 'moe_scaled' or featurizer_kind == 'moe_informative':
        numberOfColumns =   317
    elif featurizer_kind == 'mordred_raw' :
        numberOfColumns = 1613
    elif featurizer_kind == 'mordred_filtered':
        numberOfColumns = 1555
    elif featurizer_kind == 'rdkit_raw':
        numberOfColumns = 200
    elif n_cols != None:
        numberOfColumns = n_cols
    else:
        raise ValueError('Entered featurizer kind does not exist. Please use a feature kind already listed in atomsci/ddm/data/descriptor_sets_sources_by_descr_type.csv')
    #***************************************
    
    
        
    #***************************************
    i = 0
    while i <= numberOfColumns + 1:
        j = i +1
        tempFeaturizer = _shuffle_columns(tempFeaturizer, j, columns,feature_list_grid)
        pred_df = pfm.predict_from_model_file(model_path=model_path, input_df = tempFeaturizer, id_col = id_col, smiles_col = smiles_col, response_col = response_col, is_featurized=True)
        _making_dataframe(model_type, pred_df, auc_roc_list_grid, r2_list_grid,response_col)
        tempFeaturizer = input_df.copy()
        ##The temp featurizer is reset when it copies the initial dataset.
        print(f'Finished testing feature {i}/{numberOfColumns}')
        i+=1
    #***************************************
    
    
    
    
    #***************************************
    if model_type == 'regression':
        perf_df = pd.DataFrame({"Features": feature_list_grid, "r2_score": r2_list_grid})
        return perf_df
    elif model_type == 'classification':
        perf_df = pd.DataFrame({"Features": feature_list_grid, "auc_roc_score": auc_roc_list_grid})
        return perf_df
    #***************************************


        
        
