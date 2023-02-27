import pandas as pd
import re

# read Windows Slides and Intents csv to create Panda Dataframes
data = pd.read_csv('Windows_Block_Masterv7.csv', sep= None)
data1 = pd.read_csv('Intents.csv', sep= None)

# remove non-numerics from Slide column values
def remove_non_numerics(s):
    return re.sub('[^0-9]+', '', s)

data.iloc[:, 0] = data.iloc[:, 0].apply(remove_non_numerics)

# insert new column names Slide and Data
data.insert(loc=0,
          column='Slide',
          value=data.iloc[:, 0])

data.insert(loc=1,
          column='Data',
          value=data.iloc[:, 2])

# handle parsed Notes and URL References which have 4 digit Slide numbers
data['Slide'] = data['Slide'].str.slice_replace(3,4,'')

# convert slide values to integers
data['Slide'] = pd.to_numeric(data['Slide'])

data = data[['Slide','Data']]

# group all slide data into 1 row value per slide number
data = data.groupby(['Slide'])['Data'].apply(', '.join).reset_index()

# remove unnecessary slides
#print(data2['Slides'].str.extract("(\d+):(\d+)").stack().reset_index(1, drop=True))
intervals = pd.IntervalIndex.from_tuples([(1, 19), (94, 94), (151, 151), (187, 187), (243, 243), (286, 286)])
data = data.loc[pd.cut(data.Slide, intervals, include_lowest=True).isna()]

# merge Windows Slides with Intents
data = data.merge(data1, left_on='Slide', right_on='Slides')
data = data.loc[:, ['Intent',
                    'Data',
                    'Slide']]

#print file
data2 = data.to_csv('333TRS_Dataset.csv', sep=',', index=False)

# group all Intent data into 1 row value
data2 = data.groupby(['Intent'])['Data'].apply(', '.join).reset_index()

# print file
data3 = data2.to_csv('Windows_Intents.csv', sep=',', index=False)