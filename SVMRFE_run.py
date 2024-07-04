
import pandas as pd
import numpy as np
import argparse
import torch
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--path1', '-p1', type=str, required=True, help='The first omics file name.')
    parser.add_argument('--path2', '-p2', type=str, required=True, help='The second omics file name.')
    parser.add_argument('--path3', '-p3', type=str, required=True, help='The third omics file name.')
    args = parser.parse_args()

    # read data
    omics_data1 = pd.read_csv(args.path1, header=0, index_col=None)
    omics_data2 = pd.read_csv(args.path2, header=0, index_col=None)
    omics_data3 = pd.read_csv(args.path3, header=0, index_col=None)

    # set random seed
    setup_seed(args.seed)

    omics_data1.rename(columns={omics_data1.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data2.rename(columns={omics_data2.columns.tolist()[0]: 'Sample'}, inplace=True)
    omics_data3.rename(columns={omics_data3.columns.tolist()[0]: 'Sample'}, inplace=True)

    omics_data1.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data2.sort_values(by='Sample', ascending=True, inplace=True)
    omics_data3.sort_values(by='Sample', ascending=True, inplace=True)

    # name of sample
    sample_name = omics_data1['Sample'].tolist()
 
    # Get target value, y from sample class dataset
    sample_label = pd.read_csv('data/sample_classes.csv',header=0,index_col=None)
    
    
    # Feature selection for 1st omics (Transcriptomics)
    # Get X and Y values
    X_omics1 = omics_data1.iloc[:,1:]
    Y_omics1 =  sample_label.iloc[:,1].values

    # Initialize an SVM model with a linear kernel
    estimator = SVR(kernel='linear')

    # Get the feature importance or weight
    estimator.fit(X_omics1, Y_omics1)
    features_importance = pd.DataFrame({'Columns': X_omics1.columns, 'Weight':estimator.coef_.flatten()})
    print("Features Importance: \n",features_importance)

    # Apply RFE to select the top 550 features
    selector = RFE(estimator,n_features_to_select=550, step=5)

    # Train model
    selector.fit( X_omics1,Y_omics1)

    # Get selected features list
    features_selected = pd.DataFrame({'Columns':X_omics1.columns, 'Selected':selector.support_})
    print("\nSelected Features: \n",features_selected)

    # Get features ranking list
    features_rank = pd.DataFrame({'Columns': X_omics1.columns, 'Ranking': selector.ranking_})
    print("\nFeatures Ranking: \n",features_rank)

    # Get unselected features list
    features_unselected = X_omics1.columns[np.logical_not(selector.get_support())]
    print("\nUnselected Features: \n", features_unselected)

    # Test and evaluate model
    print("\nSVM-RFE Model Performance based on Transcriptomics Data")
    print("Coefficient of determination (R^2): ",selector.score(X_omics1,Y_omics1))

    # Put selected features in dataframe with sample name
    selected_features1 = X_omics1.iloc[:, selector.support_]
    pd_selected_features1 = pd.DataFrame(selected_features1)
    pd_selected_features1.insert(0, 'Sample', sample_name)
    print("\nselected feature from 1st omics\n")
    print(pd_selected_features1)
   
    
    # Feature selection for 2nd omics
    # Get X and Y values
    X_omics2 =  omics_data2.iloc[:,1:]
    Y_omics2 = sample_label.iloc[:,1].values

    # Initialize an SVM model with a linear kernel
    estimator = SVR(kernel='linear')

    # Get the feature importance or weight
    estimator.fit(X_omics2, Y_omics2)
    features_importance = pd.DataFrame({'Columns': X_omics2.columns, 'Weight':estimator.coef_.flatten()})
    print("Features Importance: \n",features_importance)

    # Apply RFE to select the top 550 features
    selector = RFE(estimator,n_features_to_select=550, step=5)

    # Train model
    selector.fit(X_omics2, Y_omics2)

    # Get selected features list
    features_selected = pd.DataFrame({'Columns':X_omics2.columns, 'Selected':selector.support_})
    print("\nSelected Features: \n",features_selected)

    # Get features ranking list
    features_rank = pd.DataFrame({'Columns': X_omics2.columns, 'Ranking': selector.ranking_})
    print("\nFeatures Ranking: \n",features_rank)

    # Get unselected features list
    features_unselected = X_omics2.columns[np.logical_not(selector.get_support())]
    print("\nUnselected Features: \n", features_unselected)

    # Test and evaluate model
    print("\nSVM-RFE Model Performance based on Genomics Data")
    print("Coefficient of determination (R^2): ",selector.score(X_omics2,Y_omics2))

    # Get the selected features
    selected_features2 = X_omics2.iloc[:, selector.support_]
    pd_selected_features2 = pd.DataFrame(selected_features2).astype(float)
    pd_selected_features2.insert(0, 'Sample', sample_name)
    print("\nselected feature from 2nd omics\n")
    print(pd_selected_features2)
    
    
    # Feature selection for 3rd omics
    # Get X and Y values
    X_omics3 = omics_data3.iloc[:,1:]
    Y_omics3 = sample_label.iloc[:,1].values

    # Initialize an SVM model with a linear kernel
    estimator = SVR(kernel='linear')

    # Get the feature importance or weight
    estimator.fit(X_omics3, Y_omics3)
    features_importance = pd.DataFrame({'Columns': X_omics3.columns, 'Weight':estimator.coef_.flatten()})
    print("Features Importance: \n",features_importance)

    # Apply RFE to select the top 200 features
    selector = RFE(estimator,n_features_to_select=200, step=5)

    # Train model
    selector.fit(X_omics3, Y_omics3)

    # Get selected features list
    features_selected = pd.DataFrame({'Columns':X_omics3.columns, 'Selected':selector.support_})
    print("\nSelected Features: \n",features_selected)

    # Get features ranking list
    features_rank = pd.DataFrame({'Columns': X_omics3.columns, 'Ranking': selector.ranking_})
    print("\nFeatures Ranking: \n",features_rank)

    # Get unselected features list
    features_unselected = X_omics3.columns[np.logical_not(selector.get_support())]
    print("\nUnselected Features: \n", features_unselected)

    # Test and evaluate model
    print("\nSVM-RFE Model Performance based on Proteomics Data")
    print("Coefficient of determination (R^2): ",selector.score(X_omics3,Y_omics3))

    # Get the selected features
    selected_features3 = X_omics3.iloc[:, selector.support_]
    pd_selected_features3 = pd.DataFrame(selected_features3)
    pd_selected_features3.insert(0, 'Sample', sample_name)
    print("\nselected feature from 3rd omics\n")
    print(pd_selected_features3)

    # Merge all selected features
    Merge_data = pd.merge(pd_selected_features1, pd_selected_features2,on='Sample', how='inner')
    Merge_data = pd.merge(Merge_data, pd_selected_features3, on='Sample', how='inner')
    print(Merge_data)
    
    # Output merged result into a  CSV file
    Merge_data.to_csv('result/latent_data_1300_target.csv', header=True, index=False)
    print('Success! Features Selection results can be seen in result file.')
