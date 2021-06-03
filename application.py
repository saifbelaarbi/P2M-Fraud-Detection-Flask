from flask import *
import joblib
import json
import pandas as pd
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/results/<percentage>')
def results(percentage):
    return percentage


@app.route('/data', methods=['POST'])
def data():
    form_data = request.form

    model = joblib.load('./data/model_lgb.pkl')

    ss_area = joblib.load('./data/ss_area.bin')
    ss_ha = joblib.load('./data/ss_ha.bin')
    ss_hp = joblib.load('./data/ss_hp.bin')
    ss_pro = joblib.load('./data/ss_pro.bin')

    with open("./data/res_area.txt", "r") as fp:
        res_area = json.load(fp)
    with open('./data/res_hp.txt', 'r') as fp:
        res_hp = json.load(fp)
    with open('./data/res_ha.txt', 'r') as fp:
        res_ha = json.load(fp)
    with open('./data/res_pro.txt', 'r') as fp:
        res_pro = json.load(fp)
    with open('./data/inp_area.txt', 'r') as fp:
        inp_area = json.load(fp)
    with open('./data/inp_hp.txt', 'r') as fp:
        inp_hp = json.load(fp)
    with open('./data/inp_ha.txt', 'r') as fp:
        inp_ha = json.load(fp)
    with open('./data/inp_pro.txt', 'r') as fp:
        inp_pro = json.load(fp)
    inpt = pd.DataFrame(columns=['Age', 'Gender', 'Marital_Status', 'Sum_Insured', 'Policies_Revenue',
                                 'Broker_ID', 'Kind_Of_Loss', 'Policy_Holder_Province',
                                 'Policy_Holder_Area', 'Province', 'Area', 'bodily_injuries',
                                 'witnesses', 'police_report_available', 'total_claim_amount',
                                 'year_of_birth', 'month_of_birth', 'day_of_birth', 'year_of_ps',
                                 'month_of_ps', 'day_of_ps', 'year_of_pe', 'month_of_pe', 'day_of_pe',
                                 'year_of_Date_Of_Loss', 'month_of_Date_Of_Loss', 'day_of_Date_Of_Loss',
                                 'year_of_Date_Of_Claim', 'month_of_Date_Of_Claim',
                                 'day_of_Date_Of_Claim'])
    inpt.loc[0, 'Age'] = int(form_data['Age'])
    inpt.loc[0, 'Sum_Insured'] = int(form_data['Sum_Insured'])
    inpt.loc[0, 'Policies_Revenue'] = int(form_data['Policies_Revenue'])
    inpt.loc[0, 'Broker_ID'] = int(form_data['Broker_ID'])
    inpt.loc[0, 'Kind_Of_Loss'] = int(form_data['Kind_Of_Loss'])
    inpt.loc[0, 'bodily_injuries'] = int(form_data['bodily_injuries'])
    inpt.loc[0, 'witnesses'] = int(form_data['witnesses'])
    inpt.loc[0, 'total_claim_amount'] = int(form_data['total_claim_amount'])
    Birthday = pd.to_datetime(form_data['Birthday'], format="%Y-%m-%d")
    inpt.loc[0, 'year_of_birth'] = Birthday.year
    inpt.loc[0, 'month_of_birth'] = Birthday.month
    inpt.loc[0, 'day_of_birth'] = Birthday.day
    dateclaim = pd.to_datetime(form_data['Date_Of_Claim'], format="%Y-%m-%d")
    inpt.loc[0, 'year_of_Date_Of_Claim'] = dateclaim.year
    inpt.loc[0, 'month_of_Date_Of_Claim'] = dateclaim.month
    inpt.loc[0, 'day_of_Date_Of_Claim'] = dateclaim.day
    dateloss = pd.to_datetime(form_data['Date_Of_Loss'], format="%Y-%m-%d")
    inpt.loc[0, 'year_of_Date_Of_Loss'] = dateloss.year
    inpt.loc[0, 'month_of_Date_Of_Loss'] = dateloss.month
    inpt.loc[0, 'day_of_Date_Of_Loss'] = dateloss.day
    ofpe = pd.to_datetime(form_data['pe'], format="%Y-%m-%d")
    inpt.loc[0, 'year_of_pe'] = ofpe.year
    inpt.loc[0, 'month_of_pe'] = ofpe.month
    inpt.loc[0, 'day_of_pe'] = ofpe.day
    ofps = pd.to_datetime(form_data['ps'], format="%Y-%m-%d")
    inpt.loc[0, 'year_of_ps'] = ofps.year
    inpt.loc[0, 'month_of_ps'] = ofps.month
    inpt.loc[0, 'day_of_ps'] = ofps.day

    inpt.loc[0, 'police_report_available'] = form_data['police_report_available']
    inpt['police_report_available'] = inpt['police_report_available'].map({'Yes': 1, 'No': 0}).astype(int)
    inpt.loc[0,'Gender'] = form_data['Gender']
    inpt['Gender'] = inpt['Gender'].map({'Male': 1, 'Female': 2, 'Fluid': 3, 'Other': 4}).astype(int)

    inpt.loc[0,'Marital_Status'] = form_data['Marital_Status']
    inpt['Marital_Status'] = inpt['Marital_Status'].map({'Single': 0,'Married': 1, 'Widowed': 2, 'Engaged': 3, 'Divorced': 4}).astype(int)

    inpt.loc[0,'Policy_Holder_Province'] = form_data['Policy_Holder_Province']
    inpt.loc[0,'Policy_Holder_Area'] = form_data['Holder area']
    inpt.loc[0,'Province'] = form_data['Province']
    inpt.loc[0,'Area'] = form_data['Area']

    
    inpt['Policy_Holder_Area'] = inpt['Policy_Holder_Area'].replace(inp_ha, res_ha)
    inpt.loc[0,'Policy_Holder_Area'] =int(inpt.loc[0,'Policy_Holder_Area'])
    inpt['Policy_Holder_Area'] = ss_ha.fit_transform(inpt.Policy_Holder_Area.values.reshape(-1, 1))

    inpt['Policy_Holder_Province'] = inpt['Policy_Holder_Province'].replace(inp_hp, res_hp)
    inpt['Policy_Holder_Province'] = ss_hp.fit_transform(inpt.Policy_Holder_Area.values.reshape(-1, 1))

    inpt['Province'] = inpt['Province'].replace(inp_pro, res_pro)
    inpt['Province'] = ss_pro.fit_transform(inpt.Policy_Holder_Area.values.reshape(-1, 1))

    inpt['Area'] = inpt['Area'].replace(inp_area, res_area)
    inpt['Area'] = ss_area.fit_transform(inpt.Policy_Holder_Area.values.reshape(-1, 1))
    inpt['Gender']=inpt['Gender'].astype("float64")
    inpt['Marital_Status']=inpt['Marital_Status'].astype("float64")
    for e in ['total_claim_amount','year_of_birth', 'month_of_birth', 'day_of_birth', 'year_of_ps','month_of_ps', 'day_of_ps', 'year_of_pe', 'month_of_pe', 'day_of_pe',
                                 'year_of_Date_Of_Loss', 'month_of_Date_Of_Loss', 'day_of_Date_Of_Loss',
                                 'year_of_Date_Of_Claim', 'month_of_Date_Of_Claim',
                                 'day_of_Date_Of_Claim','witnesses','bodily_injuries','Kind_Of_Loss','Broker_ID','Policies_Revenue','Sum_Insured','Age']:
        inpt[e]=inpt[e].astype('int64')


    proba= 10000*model.predict(inpt)[0]
    proba = int(proba)
    results = "Fraudulent" if proba>5000 else "Not Fraudulent"

    return render_template('output.html', results=results, proba=proba/100)

