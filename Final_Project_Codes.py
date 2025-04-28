#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 11:28:25 2025

@author: roland
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping




# Set the directory path (replace with your desired path)
directory = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Final Project'

# Change the working directory
os.chdir(directory)

# Verify that the working directory has changed
print("Current working directory:", os.getcwd())




##============== DATA IMPORTATION  ==============
# Loading the dataset
data_file = '/Users/roland/Desktop/UCF LIBRARY/DATA SCIENCE 2/Final Project/hotel_bookings.csv'
Data = pd.read_csv(data_file)
Data.head()

Data.shape

## Checking for Missing Values
Data.isnull().sum()
Data.describe()

missing_data=Data.isnull().sum()
missing_data_cols=missing_data[missing_data>0]
missing_data_cols

missing_percentage = (missing_data_cols / len(Data)) * 100
missing_percentage



#### Handling Missing values

Data['country'].fillna('Unknown', inplace=True)  ## Replacing NA with unknown
Data['children'].fillna(0, inplace=True) ## Replacing NA with no kids





#========================= SOME EDA ===========================


# Plot cancellation distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='is_canceled', data=Data, palette='Set2')
plt.title('Cancellation Distribution')
plt.xlabel('Is Canceled')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
plt.show()



# Cancellation Rate with Percentages
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='is_canceled', data=Data, palette='winter')
total = len(Data)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.2f}%'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=10)
plt.title('Cancellation Distribution')
plt.xlabel('')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
plt.tight_layout()
plt.show()



## Guests by countries
guests_by_country = Data[Data['is_canceled'] == 0]['country'].value_counts().reset_index()
guests_by_country.columns = ['country', 'No of guests']
guests_by_country

top_ten_countries=guests_by_country.head(10)
top_ten_countries

plt.figure(figsize=(12, 8))
sns.barplot(x=top_ten_countries['country'], y=top_ten_countries['No of guests'], palette='viridis')
plt.title("Top 10 Countries with Highest Number of Guests")
plt.xlabel("Country")
plt.ylabel("Number of Guests")
plt.xticks(rotation=45)
plt.tight_layout()  
plt.show()



## Reservation cancelation by countries
cancel_by_country=Data[Data['is_canceled'] == 1]['country'].value_counts().reset_index()
cancel_by_country.columns = ['country', 'cancellation_num']
cancel_by_country

top_10_canceled=cancel_by_country.head(10)
top_10_canceled

total_canceled=cancel_by_country['cancellation_num'].sum()
top_10_canceled['percentage']=top_10_canceled['cancellation_num']/total_canceled*100
top_10_canceled


other_sum = cancel_by_country.loc[~cancel_by_country['country'].isin(top_10_canceled['country'])]['cancellation_num'].sum()

# Create a new DataFrame for the 'Other' category
other_data = pd.DataFrame([{'country': 'Other', 'cancellation_num': other_sum, 'percentage': other_sum / total_canceled * 100}])

# Use pd.concat to append the new row
top_10_canceled = pd.concat([top_10_canceled, other_data], ignore_index=True)


plt.figure(figsize=(10, 10))  
explode = [0.1] * len(top_10_canceled)  

plt.pie(
    top_10_canceled['percentage'], 
    labels=top_10_canceled['country'], 
    autopct='%1.1f%%', 
    startangle=140,
    labeldistance=1.1,  
    pctdistance=0.85,   
    wedgeprops={'edgecolor': 'black'},  
    explode=explode,    
)

plt.title('Top 10 Countries by Percentage of Cancellation')
plt.tight_layout()
plt.show()






df = Data.copy()


### Reservations and Cancellations Trend Over Time
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], format='%Y-%m-%d')
df['year_month'] = df['reservation_status_date'].dt.to_period('M')
y_m_counts = df.groupby('year_month').size() 
y_m_cancellations = df[df['is_canceled'] == 1].groupby('year_month').size()  # 取消数
plt.figure(figsize=(12, 6))
plt.plot(y_m_counts.index.astype(str), y_m_counts, label='Total Reservations', marker='o')
plt.plot(y_m_cancellations.index.astype(str),y_m_cancellations, label='Cancellations', marker='o')
plt.title('Reservations and Cancellations Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()




## Reservations and Cancellations Trend Over Time
if not pd.api.types.is_datetime64_any_dtype(df['reservation_status_date']):
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], format='%Y-%m-%d')

df['year_month'] = df['reservation_status_date'].dt.to_period('M')
agg_counts = df.groupby(['year_month', 'is_canceled']).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
plt.plot(agg_counts.index.astype(str), agg_counts[0], label='Total Reservations', marker='o')
plt.plot(agg_counts.index.astype(str), agg_counts[1], label='Cancellations', marker='o')
plt.title('Reservations and Cancellations Trend Over Time')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()






### Cancellation Percentage by Hotel Type
sns.set_theme(style="white")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='hotel', hue='is_canceled', data=df, palette='Set2')
ax.set_title('Cancellation Percentage by Hotel Type', fontsize=16, fontweight='bold')
ax.set_xlabel('Hotel Type', fontsize=12)
ax.set_ylabel('Percentage of Bookings', fontsize=12)

# Calculate total counts for percentages
total_counts = df.groupby('hotel').size()
for container in ax.containers:
    labels = []
    for bar in container:
        height = bar.get_height()
        hotel_type = bar.get_x() + bar.get_width() / 2
        hotel_index = int(round(hotel_type))  # map x position back to index
        hotel_name = ax.get_xticklabels()[hotel_index].get_text()
        total = total_counts[hotel_name]
        percentage = (height / total) * 100 if total > 0 else 0
        labels.append(f'{percentage:.1f}%')
    ax.bar_label(container, labels=labels, padding=3, fontsize=10)
plt.tight_layout()
plt.show()




### HEATMAP
channel_cancellation_rate = df.groupby(['hotel', 'distribution_channel']).agg({'is_canceled': 'mean'}).reset_index()
pivot_table = channel_cancellation_rate.pivot(index='distribution_channel', columns='hotel', values='is_canceled')
plt.figure(figsize=(10, 7))
sns.set_theme(style="white")
ax = sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Cancellation Rate'}
)

ax.set_title('Cancellation Rate by Distribution Channel and Hotel Type', fontsize=18, fontweight='bold', pad=15)
ax.set_xlabel('Hotel Type', fontsize=14)
ax.set_ylabel('Distribution Channel', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
plt.tight_layout()
plt.show()












#========================== MODELLING =================================

### Data Preprocessing
categorical_features = ['hotel', 'arrival_date_month', 'arrival_date_year', 'meal', 'country', 'market_segment', 'distribution_channel',
                        'is_repeated_guest', 'reserved_room_type', 'deposit_type', 'customer_type']

numerical_features = ['lead_time', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 
                      'babies', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 
                      'adr', 'total_of_special_requests']

# OneHotEncode categoricals
encoder = OneHotEncoder(sparse_output=False)
encoded_cats = encoder.fit_transform(Data[categorical_features])

# Normalize numerical features
scaler = MinMaxScaler()
scaled_nums = scaler.fit_transform(Data[numerical_features])

# Concatenate all features
X = np.hstack((scaled_nums, encoded_cats))
y = Data['is_canceled'].values

# Train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)



### Modelling
# Model Architectures

input_shape = X_train.shape[1]

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Feedforward Neural Network (NN) ---
def create_nn_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# --- 1D Convolutional Neural Network (CNN) ---
def create_cnn_model():
    model = Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# --- Long Short-Term Memory (LSTM) ---
def create_lstm_model():
    model = Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# --- Gated Recurrent Unit (GRU) ---
def create_gru_model():
    model = Sequential([
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        GRU(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



# Train and Evaluate Each Model
models = {
    "NN": create_nn_model(),
    "CNN": create_cnn_model(),
    "LSTM": create_lstm_model(),
    "GRU": create_gru_model()
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC-ROC": auc
    }



# Display Results
print("\n\nModel Comparison Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")


### Tabulate Model Results
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
results_df




### Ploting results
models = ['NN', 'CNN', 'LSTM', 'GRU']
accuracy = [0.8502, 0.8457, 0.7483, 0.8400]
precision = [0.8437, 0.8227, 0.9894, 0.8097]
recall = [0.7309, 0.7436, 0.3239, 0.7425]
f1_score = [0.7833, 0.7812, 0.4881, 0.7747]
auc_roc = [0.9273, 0.9212, 0.7793, 0.9125]
bar_width = 0.15
x = np.arange(len(models))
plt.figure(figsize=(12, 7))
plt.bar(x - 2*bar_width, accuracy, width=bar_width, label='Accuracy', color='black')
plt.bar(x - bar_width, precision, width=bar_width, label='Precision', color='green')
plt.bar(x, recall, width=bar_width, label='Recall', color='blue')
plt.bar(x + bar_width, f1_score, width=bar_width, label='F1-Score', color='red')
plt.bar(x + 2*bar_width, auc_roc, width=bar_width, label='AUC-ROC', color='gold')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.title('Performance Metrics Comparison of Deep Learning Models', fontsize=16)
plt.xticks(x, models)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()







