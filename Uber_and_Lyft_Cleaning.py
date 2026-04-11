import pandas as pd

#Load dataset
df = pd.read_csv('cab_rides.csv')

#Drop product_id and id columns
df = df.drop('product_id',axis=1)
df = df.drop('id',axis=1)

#Drop rows missing price
df_clean = df.dropna(subset=['price'])

#Unix time_stamp column is changed from miliseconds to seconds 
#to prevent error when converting
df_clean['time_stamp'] = df_clean['time_stamp'] // 1000  

#Converting unix timestamp to mm:dd:yyyy hh:mm
df_clean['time_stamp'] = pd.to_datetime(df_clean['time_stamp'], unit='s')
df_clean['year'] = df_clean['time_stamp'].dt.year
df_clean['month'] = df_clean['time_stamp'].dt.month
df_clean['dayofweek'] = df_clean['time_stamp'].dt.dayofweek
df_clean['hour'] = df_clean['time_stamp'].dt.hour

#Save as new csv
df_clean.to_csv('cleaned_cab_rides.csv', index=False)