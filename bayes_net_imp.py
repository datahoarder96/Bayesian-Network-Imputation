import glob
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
os.chdir("/home/thessaron/Downloads/Datasets")

#initializing File Count
fcount = 0

# Reading List of Datasets
abs_err_dt = pd.read_excel('/home/thessaron/Downloads/List of Datasets.xlsx',header=None)


filenames = glob.glob('*.xlsx')

#Loop through Files
for filename in filenames:
    print('Dataset: '+filename)
    originalDataset = pd.read_excel(filename,header=None)
    dataset = pd.read_excel(filename,header=None)
    ImputedDataset = pd.read_excel(filename,header=None)

    # Initiaizing Data
    model = GaussianNB()
    d = defaultdict(LabelEncoder)
    imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    ImputedDataset = imp.fit_transform(ImputedDataset)
    dataset = dataset.dropna()

    #Read Original Dataset
    if('TTTEG' in filename):
        original = pd.read_excel('/home/thessaron/Downloads/Dataset_complete/TTTTEG.xlsx',header=None)
    if('C4' in filename):
        original = pd.read_excel('/home/thessaron/Downloads/Dataset_complete/C4.xlsx',header=None)
    if('HOV' in filename):
        original = pd.read_excel('/home/thessaron/Downloads/Dataset_complete/HOV.xlsx',header=None)
    if('MUSH' in filename):
        original = pd.read_excel('/home/thessaron/Downloads/Dataset_complete/MUSH.xlsx',header=None)
    if('Splice' in filename):
        original = pd.read_excel('/home/thessaron/Downloads/Dataset_complete/Splice.xlsx',header=None)
    
    #Encoding Datasets
    dataset = dataset.apply(lambda x: d[x.name].fit_transform(x))
    ImputedDataset = pd.DataFrame(ImputedDataset).apply(lambda x:d[x.name].fit_transform(x))

    # Looping through Columns of dataset
    for i in range(len(originalDataset.columns)-1):

    # Spliting input and output
        X_Train_df = dataset.iloc[:,dataset.columns!= i]
        y_Train_df = dataset.iloc[:,lambda dataset: [i]]
        X_Train = X_Train_df.values
        y_Train = y_Train_df.values

    # Training using Encoded Data
        model.fit(X_Train,y_Train)

    # Encoding data to impute
        X_temp = pd.DataFrame(ImputedDataset)
        X_temp = X_temp.iloc[:,X_temp.columns!= i]
    
    # Imputing Data
        y_imp = model.predict(X_temp)
        y_impdf = pd.DataFrame(y_imp)

    # Copying Imputed Data into the Dataframe
        ImputedDataset_df = pd.DataFrame(ImputedDataset)
        for j in range(len(ImputedDataset_df.index)):
            if pd.isnull(originalDataset.iloc[j,i]):
                ImputedDataset[i][j] = y_imp[j]
        ImputedDataset_df = pd.DataFrame(ImputedDataset)

            
# Deleting Temp Dataframes
        del y_impdf
        del y_Train_df
        del X_Train_df
        del X_temp
        del y_imp
        del X_Train
        del y_Train

# Decoding Imputed Dataset
    Imputed_Dataset = ImputedDataset_df.apply(lambda x:d[x.name].inverse_transform(x))

# Writing Imputed Dataset
    filen = "/home/thessaron/Downloads/Output/"+filename
    Imputed_Dataset.to_excel(filen,index=False,header=False)
    imputed = pd.read_excel(filen,header=None)
    test = pd.read_excel(filename,header=None)
    counter = 0
    result = 0
    for i in range(len(original.index)):
        for j in range(len(original.columns)):
            if pd.isnull(test.iloc[i,j]):
                if original.iloc[i,j] == imputed.iloc[i,j]:
                    counter = counter + 1
                    result = result + 1
                else:
                    result = result + 1

# Writing AE
    AE = counter/result
    abs_err_dt.loc[fcount,0] = filename
    abs_err_dt.loc[fcount,1] = AE
    print(AE)

#Incrementing File Counter
    fcount = fcount + 1

# writing AE output
abs_err_dt.to_excel("/home/thessaron/Downloads/AE/AEOUT.xlsx",index=False,header=False)