import numpy as np
import pandas as pd
from string import Formatter
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import joblib 
import time
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s', inputtype='timedelta'):
    """Convert a datetime.timedelta object or a regular number to a custom-
    formatted string, just like the stftime() method does for datetime.datetime
    objects.
    The fmt argument allows custom formatting to be specified.  Fields can
    include seconds, minutes, hours, days, and weeks.  Each field is optional.
    Some examples:
        '{D:02}d {H:02}h {M:02}m {S:02}s' --> '05d 08h 04m 02s' (default)
        '{W}w {D}d {H}:{M:02}:{S:02}'     --> '4w 5d 8:04:02'
        '{D:2}d {H:2}:{M:02}:{S:02}'      --> ' 5d  8:04:02'
        '{H}h {S}s'                       --> '72h 800s'
    The inputtype argument allows tdelta to be a regular number instead of the
    default, which is a datetime.timedelta object.  Valid inputtype strings:
        's', 'seconds',
        'm', 'minutes',
        'h', 'hours',
        'd', 'days',
        'w', 'weeks'
    """
    # Convert tdelta to integer seconds.
    if inputtype == 'timedelta':
        remainder = int(tdelta.total_seconds())
    elif inputtype in ['s', 'seconds']:
        remainder = int(tdelta)
    elif inputtype in ['m', 'minutes']:
        remainder = int(tdelta)*60
    elif inputtype in ['h', 'hours']:
        remainder = int(tdelta)*3600
    elif inputtype in ['d', 'days']:
        remainder = int(tdelta)*86400
    elif inputtype in ['w', 'weeks']:
        remainder = int(tdelta)*604800
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)
questionDF = pd.read_csv('/Users/lequochuy/Documents/Research/code paper2//Question-bank-KTLT.csv', sep=';')
reportsDF = pd.read_csv('/Users/lequochuy/Documents/Research/code paper2//new_reports.csv', sep=';', dtype={'ID number': str, 'questionID': str})
questionDF = questionDF.rename(columns={'bkel-link,': 'bkel-link'})
questionDF = questionDF.rename(columns={'Pre/In/Post': 'type', 'Lab': 'lab'})
questionDF['question'] = questionDF['question'].astype(str)
questionDF['level'] = questionDF['level'].apply(lambda x: x[0])
questionDF['questionID'] = questionDF['bkel-link'].apply(lambda x: x[57:])
questionDF['questionID'] = questionDF['questionID'].str.rstrip(',')
processDF = questionDF.drop(columns=['question', 'question name','bkel-link'])
# labType = ['Prelab', 'Inlab', 'Postlab']
labType = ['Prelab', 'Inlab']
labList = [1,2,3,4]
labDict = {}
for lab in labList:
    for type in labType:
        labDF = processDF[processDF['lab'] == lab]
        labWithTypeDF = labDF[labDF['type'] == type]
        labSeries = labWithTypeDF['questionID']
        labDict[type+str(lab)] = labSeries
# reportsDF.head()
reportsDF = reportsDF.rename(columns={'ID number': 'studentID', 'Time taken': 'duration', 'Started on': 'startTime', 'Completed': 'endTime', 'Grade/10.00': 'score'})
# reportsDF = reportsDF.drop(columns=['Surname', 'First name','State', 'Q. 1 /10.00'])
reportsDF.info()
# Liên kết hai DataFrame thông qua cột 'c'
data = pd.merge(processDF, reportsDF, on='questionID', how='inner')
processDF = reportsDF
processDF = processDF.groupby(['studentID', 'questionID']).aggregate({'score':['max','min','count','first'],'endTime':['first','last']}).reset_index()
processDF.columns = [''.join(col).strip() for col in processDF.columns.values]
processDF['score'] = processDF['scoremax']
processDF['growth'] = processDF['scoremax']-processDF['scorefirst']
processDF['numAttempts'] = processDF['scorecount']
processDF['questionDone'] = processDF['numAttempts']
processDF['questionDone'] = 1
processDF['endTimefirst'] = pd.to_datetime(processDF['endTimefirst'], format='%d %B %Y %I:%M %p')
processDF['endTimelast'] = pd.to_datetime(processDF['endTimelast'], format='%d %B %Y %I:%M %p')
processDF['endDate']=processDF['endTimefirst'].dt.date
temp = processDF[processDF['questionID'].isin(labDict['Prelab1'])]
temp = processDF[processDF['numAttempts'] == 5]
questionStudentDF = pd.DataFrame({'studentID': reportsDF['studentID'].unique()}, columns=['studentID']+processDF['questionID'].unique().tolist())
questionStudentDF = questionStudentDF.set_index('studentID')
for index, row in processDF.iterrows():
    questionStudentDF.loc[row['studentID']][row['questionID']] = row['score']
def convert_to_timestamp(x):
    """Convert date objects to integers"""
    try:
        return time.mktime(x.to_pydatetime().timetuple())
    except ValueError:
        return None
labDF = pd.DataFrame({'studentID': reportsDF['studentID'].unique()})
for labName, labIDSeries in labDict.items():
    labScore = processDF[processDF['questionID'].isin(labIDSeries)]
    labScoreByStudent = labScore.groupby('studentID')['score'].agg(lambda x: x.sum() / labIDSeries.count()).reset_index()
    labScoreByStudent = labScoreByStudent.rename(columns={'score': labName})
    numAttempts = labScore.groupby(['studentID']).apply(lambda x: x['numAttempts'].sum()).reset_index(name=f'{labName}-attempts')
    growth = labScore.groupby(['studentID']).apply(lambda x: x['growth'].mean()).reset_index(name=f'{labName}-growths')
    questionsDone = labScore.groupby(['studentID']).apply(lambda x: x['numAttempts'].count()).reset_index(name=f'{labName}-questions')
    timeSpent = labScore.groupby(['studentID']).apply(lambda x: (x['endTimelast'].max() - x['endTimefirst'].min()).total_seconds()/3600).reset_index(name=f'{labName}-timeSpent')
    lastSubmit = labScore.groupby(['studentID']).apply(lambda x: convert_to_timestamp(x['endTimelast'].max())).reset_index(name=f'{labName}-lastSubmit')
    # Thêm 'studentID' vào danh sách cột cần gộp
    merge_cols = ['studentID', labName, f'{labName}-attempts', f'{labName}-questions', f'{labName}-growths', f'{labName}-timeSpent', f'{labName}-lastSubmit']
    # Gộp DataFrame mới vào labDF
    labDF = pd.merge(labDF, labScoreByStudent, how='outer', on='studentID')
    labDF = pd.merge(labDF, numAttempts, how='outer', on='studentID')
    labDF = pd.merge(labDF, questionsDone, how='outer', on='studentID')
    labDF = pd.merge(labDF, growth, how='outer', on='studentID')
    labDF = pd.merge(labDF, timeSpent, how='outer', on='studentID')
    labDF = pd.merge(labDF, lastSubmit, how='outer', on='studentID')
labDF.fillna(0, inplace=True)
def ranking(score):
    if score >= 8.5:
        return 4
    if score >= 7.0:
        return 3
    if score >= 5.5:
        return 2
    if score >= 4.0:
        return 1
    return 0
def passRanking(x):
    if x >= 5:
        return 1
    else:
        return 0
listlabNum = [1,2,3,4]
listq1DF = []
for labNum in listlabNum:
    prelab = f'Prelab{labNum}'
    prelabAttempts = prelab + '-attempts'
    prelabGrowth = prelab + '-growths'
    prelabQuestions = prelab + '-questions'
    inlab = f'Inlab{labNum}'
    inlabAttempts = inlab + '-attempts'
    inlabGrowth = inlab + '-growths'
    inlabQuestions = inlab + '-questions'
    q1DF = labDF[['studentID',prelab,prelabAttempts,prelabQuestions,prelabGrowth,\
                  inlab,inlabAttempts,inlabQuestions,inlabGrowth,]]
    q1DF[f'{inlab}-output'] = q1DF[inlab].apply(ranking)
    q1DF[f'{inlab}-output'] = q1DF[inlab]
    q1DF = q1DF.fillna(0)
    listq1DF.append(q1DF)
for count, q1DF in enumerate(listq1DF):
    labNum = count+1
    prelab = f'Prelab{labNum}'
    prelabAttempts = prelab + '-attempts'
    prelabGrowth = prelab + '-growths'
    prelabQuestions = prelab + '-questions'
    inlab = f'Inlab{labNum}'
    inlabAttempts = inlab + '-attempts'
    inlabGrowth = inlab + '-growths'
    inlabQuestions = inlab + '-questions'
    inlabOutput = inlab + '-output'
    q1DF.rename(columns={prelab: 'Prelab-result', prelabAttempts: 'Prelab-attempts', prelabQuestions: 'Prelab-questions',\
                                    prelabGrowth: 'Prelab-growths', inlab: 'Inlab-result', \
                                    inlabAttempts: 'Inlab-attempts', inlabQuestions: 'Inlab-questions', inlabGrowth: 'Inlab-growths', inlabOutput: 'Inlab-output'}, inplace=True)
q1New = pd.concat(listq1DF)
q1New = q1New[['Prelab-result', 'Prelab-attempts', 'Prelab-questions', 'Prelab-growths', 'Inlab-result']]
q1DF = q1DF[q1DF['Prelab-attempts'] > 5]
q1DF = q1DF[q1DF['Inlab-attempts'] > 5]
listlabNum = [1,2,3,4]
listq1DF = []
for labNum in listlabNum:
    prelab = f'Prelab{labNum}'
    prelabAttempts = prelab + '-attempts'
    prelabGrowth = prelab + '-growths'
    prelabQuestions = prelab + '-questions'
    prelabTimeSpent = prelab + '-timeSpent'
    prelabLastSubmit = prelab + '-lastSubmit'
    inlab = f'Inlab{labNum}'
    inlabAttempts = inlab + '-attempts'
    inlabGrowth = inlab + '-growths'
    inlabQuestions = inlab + '-questions'
    inlabTimeSpent = inlab + '-timeSpent'
    inlabLastSubmit = inlab + '-lastSubmit'
    q1DF = labDF[['studentID',prelab,prelabAttempts,prelabQuestions,prelabTimeSpent,prelabLastSubmit,prelabGrowth,\
                  inlab,inlabAttempts,inlabQuestions,inlabTimeSpent,inlabLastSubmit,inlabGrowth,]]
    # q1DF[f'{inlab}-output'] = q1DF[inlab].apply(ranking)
    q1DF[f'{inlab}-output'] = q1DF[inlab]
    q1DF = q1DF.fillna(0)
    listq1DF.append(q1DF)
datasetList = []
labelList = []
for count, q1DF in enumerate(listq1DF):
    labNum = count+1
    prelab = f'Prelab{labNum}'
    prelabAttempts = prelab + '-attempts'
    prelabGrowth = prelab + '-growths'
    prelabQuestions = prelab + '-questions'
    prelabTimeSpent = prelab + '-timeSpent'
    prelabLastSubmit = prelab + '-lastSubmit'
    inlab = f'Inlab{labNum}'
    inlabAttempts = inlab + '-attempts'
    inlabGrowth = inlab + '-growths'
    inlabQuestions = inlab + '-questions'
    inlabTimeSpent = inlab + '-timeSpent'
    inlabLastSubmit = inlab + '-lastSubmit'
    # print(q1DF.count())
    # dataset = q1DF.drop(columns=['studentID',f'Inlab{labNum}-output', inlab, inlabAttempts, inlabGrowth, inlabQuestions])
    dataset = q1DF.drop(columns=['studentID',f'Inlab{labNum}-output'])
    dataset['labNum'] = labNum
    label = q1DF[f'Inlab{labNum}-output']
    dataset.rename(columns={prelab: 'Prelab', prelabAttempts: 'Prelab-attempts', prelabQuestions: 'Prelab-questions',\
                                    prelabTimeSpent: 'Prelab-timeSpent',prelabLastSubmit:'Prelab-lastSubmit', prelabGrowth: 'Prelab-growths', inlab: 'Inlab', \
                                    inlabAttempts: 'Inlab-attempts', inlabQuestions: 'Inlab-questions', inlabTimeSpent: 'Inlab-timeSpent', \
                                    inlabLastSubmit: 'Inlab-lastSubmit', inlabGrowth: 'Inlab-growths'}, inplace=True)
    datasetList.append(dataset)
    labelList.append(label)
dataset = pd.concat(datasetList)
dataset.fillna(0,inplace=True)
label = pd.concat(labelList)
label.fillna(0,inplace=True)
print(labDF)
# ------------------------------------------------------------------------- ở trên là processing ------------------------------------------
# ------------------------------------------------------------------------- ở dưới là model ------------------------------------------

labDF = labDF[labDF['Prelab1-attempts']>5]
labDF = labDF[labDF['Prelab2-attempts']>5]
labDF = labDF[labDF['Prelab3-attempts']>5]
columns_to_rank = ['Inlab1','Inlab2','Inlab3']
for column in columns_to_rank:
    labDF[column] = labDF[column].apply(ranking)
X3 = labDF[['Prelab1', 'Prelab1-attempts', 'Prelab1-questions',
            'Prelab1-growths', 'Inlab1','Inlab1-attempts', 
            'Inlab1-questions', 'Inlab1-growths', 'Prelab2', 
            'Prelab2-attempts','Prelab2-questions','Prelab2-growths','Inlab2','Inlab2-attempts',
            'Inlab2-questions','Inlab2-growths','Prelab3', 'Prelab3-attempts','Prelab3-questions','Prelab3-growths']]
Y3 = labDF['Inlab3']
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, Y3, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train3_scaled = scaler.fit_transform(X_train3)
X_test3_scaled = scaler.transform(X_test3)

# Định nghĩa và huấn luyện mô hình RandomForest
best_rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=500,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features=4
)
best_rf_model.fit(X_train3_scaled, y_train3)

# Dự đoán và tính toán độ chính xác
y_pred = best_rf_model.predict(X_test3_scaled)
accuracy_Random = accuracy_score(y_test3, y_pred)
print("Accuracy for Random Forest with MinMax Scaling:", accuracy_Random)

"""
# Lưu mô hình và scaler vào file
model_filename = "best_rf_model_with_scaler.pkl"
joblib.dump(best_rf_model, model_filename)
scaler_filename = "scaler.pkl"
joblib.dump(scaler, scaler_filename)

print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")
"""


"""
# Tải mô hình và scaler từ file
model_filename = "best_rf_model_with_scaler.pkl"
scaler_filename = "scaler.pkl"

loaded_rf_model = joblib.load(model_filename)
loaded_scaler = joblib.load(scaler_filename)

# Nhập input mới từ người dùng
input_data = input("Nhập các giá trị đặc trưng, cách nhau bằng dấu phẩy: ")
input_data = [float(x) for x in input_data.split(",")]

# Chuyển đổi input thành định dạng numpy array và chuẩn hóa
input_array = np.array(input_data).reshape(1, -1)  # Đảm bảo input là một mảng 2D
input_array_scaled = loaded_scaler.transform(input_array)

# Dự đoán với mô hình đã tải
y_pred_loaded = loaded_rf_model.predict(input_array_scaled)
print("Dự đoán cho input đã nhập là:", y_pred_loaded[0])
"""
