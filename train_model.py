import joblib
import pandas as pd
import json
import os
from joblib import *
import lightgbm as lgb
import sklearn

import warnings
warnings.filterwarnings('ignore')

# class pretrained_model():


def train():
    check = ['res_hp.txt', 'ss_hp.bin', 'inp_area.txt', '.~lock.insurance.xlsx#', 
    'res_area.txt', 'inp_pro.txt', 'res_pro.txt', 'model_lgb.pkl', 'ss_ha.bin', 'ss_pro.bin', 
    'inp_hp.txt', 'ss_area.bin', 'res_ha.txt', 'inp_ha.txt', 'insurance.xlsx']
    trained = False
    print(os.listdir(os.getcwd()+'/data'))
    if sorted(os.listdir(os.getcwd()+'/data')) == sorted(check):
        print("Model is pre-trained")
        answear = input("Would you like to retrain model?:[y/n]")
        if(answear == "n"):
            print('Model Not Trained')
            trained=True
        elif(answear != "y"):
            print('please enter y or n')
            trained=True
        elif(answear == "y"):
            trained = False

    if not trained:

        print('Model is not pre-trained!')

        print('Loading Data...')

        Data = pd.read_excel('./data/insurance.xlsx')
        
        print('Feature Engineering...')
        Data['year_of_birth'] = pd.to_datetime(
            Data['Date_Of_Birth']).apply(lambda x: x.year)
        Data['month_of_birth'] = pd.to_datetime(
            Data['Date_Of_Birth']).apply(lambda x: x.month)
        Data['day_of_birth'] = pd.to_datetime(
            Data['Date_Of_Birth']).apply(lambda x: x.day)
        Data['year_of_ps'] = pd.to_datetime(
            Data['Policy_Start']).apply(lambda x: x.year)
        Data['month_of_ps'] = pd.to_datetime(
            Data['Policy_Start']).apply(lambda x: x.month)
        Data['day_of_ps'] = pd.to_datetime(
            Data['Policy_Start']).apply(lambda x: x.day)
        Data['year_of_pe'] = pd.to_datetime(
            Data['Policy_End']).apply(lambda x: x.year)
        Data['month_of_pe'] = pd.to_datetime(
            Data['Policy_End']).apply(lambda x: x.month)
        Data['day_of_pe'] = pd.to_datetime(
            Data['Policy_End']).apply(lambda x: x.day)
        Data['year_of_Date_Of_Loss'] = pd.to_datetime(
            Data['Date_Of_Loss']).apply(lambda x: x.year)
        Data['month_of_Date_Of_Loss'] = pd.to_datetime(
            Data['Date_Of_Loss']).apply(lambda x: x.month)
        Data['day_of_Date_Of_Loss'] = pd.to_datetime(
            Data['Date_Of_Loss']).apply(lambda x: x.day)
        Data['year_of_Date_Of_Claim'] = pd.to_datetime(
            Data['Date_Of_Claim']).apply(lambda x: x.year)
        Data['month_of_Date_Of_Claim'] = pd.to_datetime(
            Data['Date_Of_Claim']).apply(lambda x: x.month)
        Data['day_of_Date_Of_Claim'] = pd.to_datetime(
            Data['Date_Of_Claim']).apply(lambda x: x.day)
        Data['Gender'] = Data['Gender'].map(
            {'Male': 1, 'Female': 2, 'Fluid': 3, 'Other': 4}).astype(int)
        Data['Fraudulent_Claim'] = Data['Fraudulent_Claim'].map(
            {'T': 1, 'F': 0}).astype(int)

        pd.get_dummies(
            Data, columns=['Gender'], drop_first=True)

        to_drop = ['Policy_Holder_City','Calim_ID', 'Name', 'Surname', 'Date_Of_Birth', 'Service_Provider', 'Insured_ID', 'Policy_Holder_Street', 'City',
                'Postal_Code', 'Policy_Holder_Postal', 'Policy_Start', 'Policy_End', 'Date_Of_Loss', 'Date_Of_Claim', 'Party_Name', 'Party_Surname']
        Data.drop(to_drop, axis=1, inplace=True)

        inp_ha = tuple(Data[['Policy_Holder_Area', 'Fraudulent_Claim']].groupby(['Policy_Holder_Area'],
                                                                                as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Policy_Holder_Area'])
        res_ha = tuple(Data[['Policy_Holder_Area', 'Fraudulent_Claim']].groupby(['Policy_Holder_Area'],
                                                                                as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Fraudulent_Claim'])
        ss_ha = sklearn.preprocessing.StandardScaler()
        Data['Policy_Holder_Area'] = Data['Policy_Holder_Area'].replace(
            inp_ha, res_ha)
        Data['Policy_Holder_Area'] = ss_ha.fit_transform(
            Data.Policy_Holder_Area.values.reshape(-1, 1))

        Data[['Policy_Holder_Province', 'Fraudulent_Claim']].groupby(['Policy_Holder_Province'],
                                                                    as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)
        inp_hp = tuple(Data[['Policy_Holder_Province', 'Fraudulent_Claim']].groupby(['Policy_Holder_Province'],
                                                                                    as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Policy_Holder_Province'])
        res_hp = tuple(Data[['Policy_Holder_Province', 'Fraudulent_Claim']].groupby(['Policy_Holder_Province'],
                                                                                    as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Fraudulent_Claim'])
        ss_hp = sklearn.preprocessing.StandardScaler()
        Data['Policy_Holder_Province'] = Data['Policy_Holder_Province'].replace(
            inp_hp, res_hp)
        Data['Policy_Holder_Province'] = ss_hp.fit_transform(
            Data.Policy_Holder_Area.values.reshape(-1, 1))

        inp_pro = tuple(Data[['Province', 'Fraudulent_Claim']].groupby(['Province'],
                                                                    as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Province'])
        res_pro = tuple(Data[['Province', 'Fraudulent_Claim']].groupby(['Province'],
                                                                    as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Fraudulent_Claim'])
        ss_pro = sklearn.preprocessing.StandardScaler()
        Data['Province'] = Data['Province'].replace(inp_pro, res_pro)
        Data['Province'] = ss_pro.fit_transform(
            Data.Policy_Holder_Area.values.reshape(-1, 1))

        inp_area = tuple(Data[['Area', 'Fraudulent_Claim']].groupby(['Area'],
                                                                    as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Area'])
        res_area = tuple(Data[['Area', 'Fraudulent_Claim']].groupby(['Area'],
                                                                    as_index=False).mean().sort_values(by='Fraudulent_Claim', ascending=False)['Fraudulent_Claim'])
        ss_area = sklearn.preprocessing.StandardScaler()
        Data['Area'] = Data['Area'].replace(inp_area, res_area)
        Data['Area'] = ss_area.fit_transform(
            Data.Policy_Holder_Area.values.reshape(-1, 1))


        with open("./data/inp_ha.txt", "w") as fp:
            json.dump(inp_ha, fp)
        with open("./data/inp_hp.txt", "w") as fp:
            json.dump(inp_hp, fp)
        with open("./data/inp_pro.txt", "w") as fp:
            json.dump(inp_pro, fp)
        with open("./data/inp_area.txt", "w") as fp:
            json.dump(inp_area, fp)
        with open("./data/res_ha.txt", "w") as fp:
            json.dump(res_ha, fp)
        with open("./data/res_hp.txt", "w") as fp:
            json.dump(res_hp, fp)
        with open("./data/res_pro.txt", "w") as fp:
            json.dump(res_pro, fp)
        with open("./data/res_area.txt", "w") as fp:
            json.dump(res_area, fp)
        joblib.dump(ss_area, './data/ss_area.bin')
        joblib.dump(ss_ha, './data/ss_ha.bin')
        joblib.dump(ss_hp, './data/ss_hp.bin')
        joblib.dump(ss_pro, './data/ss_pro.bin')

        X = Data.drop(
            ['Fraudulent_Claim', 'Fraudulent_Claim_Reason'], axis=1)

        y = Data.Fraudulent_Claim
        print(X.columns)
        ##y_reason=Data.Fraudulent_Claim_Reason
        print('Model Training...')
        params = {'bagging_fraction': 0.8192059420307759,
                'boost_from_average': False,
                'feature_fraction': 0.854792829599626,
                'is_unbalance': True,
                'learning_rate': 0.8277896457253541,
                'max_bin': 56,
                'max_depth': 27,
                'metric': 'auc',
                'min_data_in_leaf': 55,
                'min_sum_hessian_in_leaf': 45.00845042447376,
                'num_leaves': 62,
                'objective': 'binary',
                'subsample': 0.4252313651032907}

        final_model = lgb.LGBMModel(**params)
        print(X.dtypes)
        final_model.fit(X, y)
        pred_proba=final_model.predict(X)
        pred = [1 if x>0.5 else 0 for x in pred_proba]
        print('Saving Model...')
        joblib.dump(final_model, './data/model_lgb.pkl')
        print('DONE.')
        


if __name__ == "__main__":
    train()
