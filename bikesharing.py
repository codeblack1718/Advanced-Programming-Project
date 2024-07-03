


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()

import io
df_hour = pd.read_csv(io.BytesIO(uploaded['hour.csv']))

df_hour.head()

df_hour.shape

df_hour.info()

df_hour.isnull().sum()

df_hour.duplicated().sum()

df_hour=df_hour.drop(columns=['instant','dteday'])

season_string ={
    1:'winter',
    2:'spring',
    3:'summer',
    4:'fall'
}
df_hour['season'] = df_hour['season'].map(season_string)
df_hour['season'].head()

weekday_string = {
    1:'monday',
    2:'tuesday',
    3:'wednesday',
    4:'thursday',
    5:'friday',
    6:'saturday',
    0:'sunday'
}
df_hour['weekday'] = df_hour['weekday'].map(weekday_string)
df_hour['weekday'].head()

df_hour['temp2'] = (df_hour['temp'])*47-8
df_hour['atemp2']=(df_hour['atemp'])*66-16
df_hour['hum2']=df_hour['hum']*100
df_hour['windspeed2']=df_hour['windspeed']*67

temp2_bins=np.linspace(df_hour['temp2'].min(),df_hour['temp2'].max(),4)
temp2_labels=['Cold','Mild','Hot']
df_hour['temp2_binned']=pd.cut(df_hour['temp2'],bins=temp2_bins,labels=temp2_labels)

hum2_bins=np.linspace(df_hour['hum2'].min(),df_hour['hum2'].max(),4)
hum2_labels=['Low','Medium','High']
df_hour['hum2_binned']=pd.cut(df_hour['hum2'],bins=hum2_bins,labels=hum2_labels)

windspeed2_bins=np.linspace(df_hour['windspeed2'].min(),df_hour['windspeed2'].max(),4)
windspeed2_labels=['Calm','Breezy','Windy']
df_hour['windspeed2_binned']=pd.cut(df_hour['windspeed2'],bins=windspeed2_bins,labels=windspeed2_labels)

hours_bins=[0,6,12,18,24]
hour_labels=['Early Morning','Morning','Afternoon','Evening']
df_hour['hour_binned']=pd.cut(df_hour['hr'],bins=hours_bins,labels=hour_labels,right=False)

cnt_bins=np.linspace(df_hour['cnt'].min(),df_hour['cnt'].max(),4)
cnt_labels=['Low','Medium','High']
df_hour['cnt_binned']=pd.cut(df_hour['cnt'],bins=cnt_bins,labels=cnt_labels)

df_hour[['casual','registered','cnt']].describe()

categorical_columns = ['season','yr','mnth','weekday','weathersit','temp2_binned','hum2_binned','windspeed2_binned','hour_binned','cnt_binned']
value_counts_dict = {}
for column in categorical_columns:
  value_counts_dict[column]=df_hour[column].value_counts()
for column,value_counts in value_counts_dict.items():
  print(f"Value counts for {column}:\n{value_counts}\n")

categorical_columns=['season','weekday','weathersit','temp2_binned','hum2_binned','windspeed2_binned','hour_binned','cnt_binned']
mode_values=df_hour[categorical_columns].mode().iloc[0]
mode_values

seasonal_data=df_hour.groupby('season')[['casual','registered']].sum().reset_index()
seasonal_data['total']=seasonal_data['casual']+seasonal_data['registered']
seasonal_data['casual_percentage']=(seasonal_data['casual']/seasonal_data['total'])*100
seasonal_data['registered_percentage']=(seasonal_data['registered']/seasonal_data['total'])*100
print(seasonal_data[['season','casual_percentage','registered_percentage']])

monthly_data=df_hour.groupby('mnth')[['casual','registered']].sum().reset_index()
monthly_data['total']=monthly_data['casual']+monthly_data['registered']
monthly_data['casual_percentage']=(monthly_data['casual']/monthly_data['total'])*100
monthly_data['registered_percentage']=(monthly_data['registered']/monthly_data['total'])*100
print(monthly_data[['mnth','casual_percentage','registered_percentage']])







df_hour.groupby('weekday')[['casual','registered']].mean().reset_index()

df_hour.groupby('hour_binned')[['cnt','casual','registered']].mean().reset_index()

numeric_columns=['temp2','atemp2','hum2','windspeed2','casual','registered','cnt']
correlation_matrix=df_hour[numeric_columns].corr()
print(correlation_matrix)
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

numeric_columns=['temp2','atemp2','hum2','windspeed2','casual','registered','cnt']
correlation_matrix=df_hour[numeric_columns].corr()
rider_correlations =correlation_matrix.loc[['cnt','casual','registered'],['temp2','atemp2','hum2','windspeed2']]
print(rider_correlations)
plt.figure(figsize=(10,8))
sns.heatmap(rider_correlations,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title('Rider Correlations with Weather for casual and registered rider')
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(data=df_hour,x='mnth',y='casual',marker='o',label='Casual_Users',color='blue')
sns.lineplot(data=df_hour,x='mnth',y='registered',marker='o',label='Registered_Users',color='red')
plt.title('Casual and Registered Users by Month')
plt.xlabel('Month')
plt.ylabel('Number of Users')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
sns.lineplot(data=df_hour,x='weekday',y='casual',marker='o',label='Casual_Users',color='blue')
sns.lineplot(data=df_hour,x='weekday',y='registered',marker='o',label='Registered_Users',color='red')
plt.xlabel('Day')
plt.ylabel('Number of Riders')
plt.title('Casual and Registered Riders by week')
plt.legend()
plt.show()

plt.figure (figsize=(10, 6))
sns.lineplot(data=df_hour, x='hr', y='casual', marker='o', label='Casual Users', color='blue')
sns.lineplot(data=df_hour, x='hr', y='registered', marker='o', label='Registered Users', color='red')
plt.xlabel('Hour')
plt.ylabel('Number of Riders')
plt.title('Casual and Registered Riders by Hour')
plt.legend()
plt.show

plt.figure(figsize=(12,6))
sns.barplot(data=df_hour,x='season',y='cnt',hue='weekday')
plt.title('Total Users By Seasons')
plt.xlabel('Season')
plt.ylabel('Total Users')
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=df_hour,x='weathersit',y='cnt',hue='weekday')
plt.title('Total Users by weather situation')
plt.xlabel('Weather Situation')
plt.ylabel('Total Users')
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=df_hour,x='holiday',y='cnt',hue='weekday')
plt.title('Total Users by Holiday')
plt.xlabel('Holiday')
plt.ylabel('Total Users')

plt.figure(figsize=(12,6))
sns.barplot(data=df_hour,x='temp2_binned',y='cnt',hue='weekday')
plt.title('Total Users by weather situation')
plt.xlabel('Weather Situation')
plt.ylabel('Total Users')
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=df_hour,x='hum2_binned',y='cnt',hue='weekday')
plt.title('Total Users by humidity')
plt.xlabel('Humidity')
plt.ylabel('Total Users')
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=df_hour,x='windspeed2_binned',y='cnt',hue='weekday')
plt.title('Total Users by wind speed')
plt.xlabel('Wind Speed')
plt.ylabel('Total Users')
plt.show()