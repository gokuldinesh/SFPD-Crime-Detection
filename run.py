import pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

def train_model(train_x,train_y):
    
    clf = RandomForestClassifier(n_estimators = 200, max_depth = 14)
    clf = clf.fit(train_x, train_y)
    
    return clf

def data_extractor(fname, train = False):
    
    data = pd.read_csv(fname)
    
    data['Dates'] = pd.to_datetime(data['Dates'])
    data['Year'] = data['Dates'].dt.year
    data['Month'] = data['Dates'].dt.month
    data['Day'] = data['Dates'].dt.day
    data['Time'] = data['Dates'].dt.hour*60 + data['Dates'].dt.minute

    data['DayOfWeek'] = pd.Categorical(data['DayOfWeek'],categories = data['DayOfWeek'].unique()).codes
    data['PdDistrict'] = pd.Categorical(data['PdDistrict'],categories = data['PdDistrict'].unique()).codes
    data['Resolution'] = pd.Categorical(data['Resolution'],categories = data['Resolution'].unique()).codes
    
    
    if(train):
        data = data.sort_values(['Category'])
        data['Category'] = pd.Categorical(data['Category'],categories = data['Category'].unique()).codes
    
    return data

def outlier_detector(data):
    
    hi_x_q = data['X'].quantile(0.75)+1.5*(data['X'].quantile(0.75)-data['X'].quantile(0.25))
    lo_x_q = data['X'].quantile(0.25)-1.5*(data['X'].quantile(0.75)-data['X'].quantile(0.25))
    hi_y_q = data['Y'].quantile(0.75)+1.5*(data['Y'].quantile(0.75)-data['Y'].quantile(0.25))
    lo_y_q = data['Y'].quantile(0.25)-1.5*(data['Y'].quantile(0.75)-data['Y'].quantile(0.25))
    data = data[data['X'] < hi_x_q]
    data = data[data['X'] > lo_x_q]
    data = data[data['Y'] < hi_y_q]
    data = data[data['Y'] > lo_y_q]
    
    return data
    
def prep_data(data, train = False):
    
    ids = data['Id'].as_matrix()

    x = [data['DayOfWeek'].as_matrix(), data['PdDistrict'].as_matrix(), data['Resolution'].as_matrix(),
         data['X'].as_matrix(), data['Y'].as_matrix(), data['Year'].as_matrix(), data['Year'].as_matrix(), 
         data['Time'].as_matrix()]
    x = np.transpose(x)
    
    if(train):
        y = data['Category'].as_matrix()
        return x, y, ids
    
    return x, ids
    
def output_data(test_y, test_ids):
    
    out = pd.DataFrame(data = test_ids, columns = ['Id'], index = None)
    out_data = pd.DataFrame(data = test_y, columns=['ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY',
                                                   'DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE',
                                                   'DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION',
                                                   'FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD',
                                                   'GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS',
                                                   'LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
                                                   'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE',
                                                   'ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE',
                                                   'SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE',
                                                   'SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT',
                                                   'WARRANTS','WEAPON LAWS'])
    
    out = out.join(out_data)
    out.to_csv(path_or_buf = 'output.csv', sep = ',', index = False)
    
train_file = sys.argv[1]
test_file = sys.argv[2]

train_x, train_y, train_ids = prep_data(outlier_detector(data_extractor(train_file, train = True)), train = True)
test_x, test_ids = prep_data(data_extractor(test_file))

print(np.shape(train_x))
print(np.shape(train_y))
print(np.shape(test_x))

print("Training model")
model = train_model(train_x,train_y)
test_y = model.predict_proba(test_x)

print(np.shape(test_y))

output_data(test_y, test_ids)