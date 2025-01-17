import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# sns.reset_orig() 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report
)
import random
from datetime import datetime, timedelta
import datetime
import time
sns.reset_defaults()


df = pd.read_csv('data_baitoan_3.csv')

X = df.drop(['labFinalScore'], axis=1)

# Target: The 'Ranking' column
y = df['labFinalScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=df['labFinalScore'],random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))



