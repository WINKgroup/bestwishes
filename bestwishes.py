from datetime import datetime

import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv (r'data.csv', names = ['date', 'wishes'], parse_dates=["date"])                     # import data
df['dayOfYear'] = df['date'].dt.dayofyear  # get day of the year for each date

svclassifier = SVC(kernel='linear')                     # setup classifier and
svclassifier.fit(df[['dayOfYear']], df['wishes'])       # train network

dates = ['2100-12-25', '2100-03-28']                    # some to data to check
days = list(map(lambda date: [ int(datetime.strptime(date, "%Y-%m-%d").strftime('%j')) ], dates ))  # feature extraction
predictions = svclassifier.predict(days)                # some to data to check

# print the output
for i in range(len(dates)):
    print('Oggi è il ' + dates[i] + ': ' + predictions[i])