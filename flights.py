import datetime
import decimal
import json
from typing import Dict
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Model, load_model
import tensorflow as tf

# @st.cache_data #(allow_output_mutation=True)
def load_pandas():
    weather_details_flights_airport_codes = pd.read_csv(f"./datasets/weather_details_flights_airport_codes.csv", index_col="Unnamed: 0")
    weather_details_flights_airport_codes.rename(columns={
    'sched_dep_time': 'scheduled departure time', 'dep_time': 'departure time',
    'dep_delay': 'departure delay', 'air_time': 'air time', 'dest':'destination', 'arr_time': 'arrival time',
    'sched_arr_time': 'scheduled arrival time', 'arr_delay':'arrival delay','eta_duration':'eta duration',
    'duration': 'travel duration', 'travel_delay': 'total travel delay', 'flight_delayed': 'is flight delayed',
    'delay_type': 'delay type', 'weather_description': 'weather description',
    'wind_direction':'wind direction', 'wind_speed':'wind speed', 'iata_code':'IATA',
    'elevation_ft': 'elevation', 'iso_country': 'country'}, inplace=True)
    
    new_forcasted_flight = ['month', 'day', 'hour','origin','scheduled departure time','departure time', 'departure delay', 'distance','destination','scheduled arrival time', 'eta duration',
                   'humidity', 'pressure', 'temperature', 'weather description', 'wind direction', 'wind speed',
                   'name', 'carrier', 'flight','airport','municipality', 'elevation']

    weather_details_flights_airport_codes.dropna(inplace=True)
    
    return weather_details_flights_airport_codes[new_forcasted_flight + ['total travel delay']]

@st.cache_data
def encode_transform_scale(curr_df:pd.DataFrame, inputs: pd.DataFrame):
    mapping = dict()
    inputs_scaler = StandardScaler()
    label_scaler = StandardScaler()
    label = LabelEncoder()

    # insert user input to end of all inputs
    curr_df = pd.concat([curr_df[['month', 'day', 'hour','origin','scheduled departure time','departure time', 'departure delay', 'distance','destination','scheduled arrival time', 'eta duration',
                'humidity', 'pressure', 'temperature', 'weather description', 'wind direction', 'wind speed',
                'name', 'carrier', 'flight','airport','municipality', 'elevation', 'total travel delay']], inputs])
    
    for column in curr_df.columns:
        if column == 'timestamp':
            curr_df[column] = pd.to_datetime(curr_df[column])

        if curr_df[column].dtype == 'object':
            curr_df[column] = label.fit_transform(curr_df[column])
            mapping[column] = dict(zip(label.transform(label.classes_), label.classes_))


    columns = list(curr_df.columns)
    label_columns = ['total travel delay']
    input_columns = list(curr_df.drop(columns=label_columns).columns)

    scaled_columns = list([ f"scl_{str(col)}" for col in columns])
    scaled_inputs = list([ f"scl_{str(col)}" for col in input_columns])
    scaled_labels = list([ f"scl_{str(col)}" for col in label_columns])

    _label_btc_mulv_scaled = label_scaler.fit_transform(curr_df[label_columns])
    _label_btc_mulv = pd.DataFrame(_label_btc_mulv_scaled, columns=scaled_labels)

    _inputs_reg_av_ds_scaled = inputs_scaler.fit_transform(curr_df[input_columns])
    _reg_av_ds_scaled = pd.DataFrame(_inputs_reg_av_ds_scaled, columns=scaled_inputs)

    scaled_inputs_df = pd.concat([_reg_av_ds_scaled,
            _label_btc_mulv],
            axis=1)

    for col in scaled_columns:
        curr_df[col] = scaled_inputs_df[col].values

    scaled_inputs_df = curr_df[scaled_columns].rename(columns={col:col.replace('scl_','') for col in scaled_columns})
    print(scaled_inputs_df.tail())
    return curr_df, scaled_inputs_df, mapping, label_scaler

# st.cache_resource
def load_pretrained_model():
    model: Model  = load_model('./models/cnn_regression_base_lr0_0001epoch50drop02-F183')
    # print(f"[+] Model Summary: {model.summary()}")
    return model

df: pd.DataFrame = load_pandas()
print(f"[+] Initial {len(df)}")

st.markdown("# TIP Finals: Model Deployment")

st.write("""
    ## Convolutional Neural Network Regression (Convo1D)
    Problem Statement: Predicting Arrival Flights Delay which is a Major problem specially with travelers with connecting flights
    """     
)
st.divider()
st.write("""
    ## Travel Plan
    """     
)

# Checkbox
if st.checkbox('Show dataframe'):
    st.dataframe(df[['total travel delay', 'origin','scheduled departure time','departure time', 'departure delay', 'distance','destination','scheduled arrival time', 'eta duration',
                   'humidity', 'pressure', 'temperature', 'weather description', 'wind direction', 'wind speed',
                   'name', 'carrier', 'flight','airport','municipality', 'elevation']])

st.write("""
    ### Primary Details
    """     
)
payload:Dict = {}

# Month Day Year
departure_date = st.date_input(label="Departure Date",
                               help="Date of Departure",
                               min_value=datetime.datetime(2023,12,30)
                               )

payload['month']= int(departure_date.month)
payload['day']= int(departure_date.day)

# scheduled departure time
scheduled_departure_time = st.time_input( label="Scheduled Departure Time", help="Flight Scheduled Departure Time")
payload['hour']= int(scheduled_departure_time.hour)
payload['scheduled departure time'] = int((scheduled_departure_time.hour*100) + (scheduled_departure_time.minute *10))

# departure time
departure_time = st.time_input(
    label="Time Departed", 
    help="Actual time the plane departed",
    value=scheduled_departure_time
    )
payload['departure time'] = int((departure_time.hour*100) + (departure_time.minute *10))
payload['departure delay'] = payload['departure time'] - payload['scheduled departure time']

# scheduled departure time
scheduled_arrival_time = st.time_input(
    label="Scheduled Arrival Time", 
    help="The Scheduled Arrival time for the destination or connecting Airport",
    value=departure_time
    )

payload['scheduled arrival time'] = int((scheduled_arrival_time.hour*100) + (scheduled_arrival_time.minute *10))

# Dropdown 
origin = st.selectbox(
    key=1,
    label='Airport Origin',
    options=df['origin'].unique(),
    )

'Airport Origin: ', origin

payload['origin'] = origin

# reg_av_ds[['name', 'carrier']].value_counts().reset_index().set_index('name').loc['United Air Lines Inc.', 'carrier']
airline_bucket = df[['name', 'carrier']].value_counts().reset_index().set_index('name')

# Dropdown
# airlines
airlines_name = st.selectbox(
    key=2,
    label='Airlines',
    options=df['name'].unique(),
    )

carrier = airline_bucket.loc[airlines_name, 'carrier']
# 'Airlines: ', airlines, 
'Carrier: ', carrier

payload['name'] = airlines_name
payload['carrier'] = carrier

# Dropdown
flight = st.selectbox(
        key=4,
        label='Flight Number',
        options=df['flight'].unique(),
    )

'Flight: ', flight
payload['flight'] = flight

st.write("""
    ### Destination
    """     
)

airport_destination_bucket = df[['destination', 'airport', 'municipality', 'elevation']].value_counts().reset_index().set_index('airport')

# Dropdown - Automap airport_destination,destination, municipality, elavation
airport = st.selectbox(
        key=5,
        label='Airport Destination',
        options=df['airport'].unique(),
    )
municipality = airport_destination_bucket.loc[airport, 'municipality']
destination = airport_destination_bucket.loc[airport, 'destination']
elevation = airport_destination_bucket.loc[airport, 'elevation']
'Municipality: ', municipality, destination
'Elevation: ',  elevation

payload['airport'] = airport
payload['municipality'] = municipality
payload['destination'] = destination
payload['elevation'] = elevation

# distance = origin + distination, eta
flight_path = df[['origin', 'destination', 'distance', 'eta duration']].set_index(keys=['destination', 'origin'])
payload['distance'] = flight_path.loc[destination, origin]['distance'].mean()
payload['eta duration'] = flight_path.loc[destination, origin]['eta duration'].mean()

st.write("""
    ## Weather Forecast
    """     
)

# Weather Forecast Described Values
described_values = df[['humidity','pressure', 'temperature', 'wind direction', 'wind speed']].describe()

humidity = st.slider(
    key=6,
    label='Humidity',
    min_value=(described_values.loc['min', 'humidity'] - described_values.loc['std', 'humidity']), 
    max_value=(described_values.loc['max', 'humidity'] + described_values.loc['std', 'humidity']), 
    value=described_values.loc['mean', 'humidity']
)

payload['humidity'] = humidity

pressure = st.slider(
    key=7,
    label='Pressure',
    min_value=(described_values.loc['min', 'pressure'] - described_values.loc['std', 'pressure']), 
    max_value=(described_values.loc['max', 'pressure'] + described_values.loc['std', 'pressure']), 
    value=described_values.loc['mean', 'pressure']
)

payload['pressure'] = pressure

temperature = st.slider(
    key=8,
    label='Temperature',
    min_value=(described_values.loc['min', 'temperature'] - described_values.loc['std', 'temperature']), 
    max_value=(described_values.loc['max', 'temperature'] + described_values.loc['std', 'temperature']), 
    value=described_values.loc['mean', 'temperature']
)

payload['temperature'] = temperature

wind_direction = st.slider(
    key=9,
    label='Wind Direction',
    min_value=(described_values.loc['min', 'wind direction'] - described_values.loc['std', 'wind direction']), 
    max_value=(described_values.loc['max', 'wind direction'] + described_values.loc['std', 'wind direction']), 
    value=described_values.loc['mean', 'wind direction']
)

payload['wind direction'] = wind_direction

wind_speed = st.slider(
    key=10,
    label='Wind Speed',
    min_value=(described_values.loc['min', 'wind speed'] - described_values.loc['std', 'wind speed']), 
    max_value=(described_values.loc['max', 'wind speed'] + described_values.loc['std', 'wind speed']), 
    value=described_values.loc['mean', 'wind speed']
)

payload['wind speed'] = wind_speed

weather_description = st.selectbox(
        key=11,
        label='Weather Description',
        options=df['weather description'].unique(),
        index=10
    )

payload['weather description'] = weather_description

st.write("""
    ## Possibility of missing a connecting flight due to arrival delay
    """     
)

left_column, right_column = st.columns(2)

if left_column.button("Your Flight Prediction", type="primary"):  
    model = load_pretrained_model()
    # payload['total travel delay'] = 0
    # test_predict = pd.DataFrame([payload])
    test_predict = pd.DataFrame([payload])
    df, scaled_df, mapping, label_scaler = encode_transform_scale(df, test_predict)
    data = scaled_df.tail(24)
    targets = data[['total travel delay']][23:]
    # print(f"[+] targets {targets}")
    data = data.drop(columns=['total travel delay'])

    test_predict = tf.keras.utils.timeseries_dataset_from_array(
        data=data, 
        targets=targets, 
        sequence_length=24,
        sequence_stride=1,
        sampling_rate=1,
        shuffle=False,
        batch_size=1)

    ypred = model.predict(test_predict)
    ypred = ypred.reshape(ypred.shape[0], 1)
    predicted = label_scaler.inverse_transform(ypred)

    delay = predicted.reshape(-1)[0]
    text_delay = ""
    if delay < 0:
        text_delay = f"Will arrive early by {delay}"
    else:
        text_delay = f"WIll arrive late in destination airport by {delay}"

    st.write(text_delay)

if right_column.button("Test Data Predict"):
    test_payload = [{'month': 9,
        'day': 16,
        'hour': 20,
        'origin': 'JFK',
        'scheduled departure time': 2001,
        'departure time': 2134.0,
        'departure delay': 93.0,
        'distance': 1826,
        'destination': 'ABQ',
        'scheduled arrival time': 2248,
        'eta duration': 247,
        'humidity': 62.0,
        'pressure': 833.0,
        'temperature': 296.014,
        'weather description': 'sky is clear',
        'wind direction': 208.0,
        'wind speed': 1.0,
        'name': 'JetBlue Airways',
        'carrier': 'B6',
        'flight': 65,
        'airport': 'Albuquerque International Sunport',
        'municipality': 'Albuquerque',
        'elevation': 5355.0,
        'total travel delay': 157.0},
        {'month': 9,
        'day': 29,
        'hour': 20,
        'origin': 'JFK',
        'scheduled departure time': 2001,
        'departure time': 2114.0,
        'departure delay': 73.0,
        'distance': 1826,
        'destination': 'ABQ',
        'scheduled arrival time': 2248,
        'eta duration': 247,
        'humidity': 19.0,
        'pressure': 835.0,
        'temperature': 294.945,
        'weather description': 'sky is clear',
        'wind direction': 277.0,
        'wind speed': 1.0,
        'name': 'JetBlue Airways',
        'carrier': 'B6',
        'flight': 65,
        'airport': 'Albuquerque International Sunport',
        'municipality': 'Albuquerque',
        'elevation': 5355.0,
        'total travel delay': 118.0}]
    
    st.write(test_payload)
    model = load_pretrained_model()

    test_predict = pd.DataFrame(test_payload)
    df, scaled_df, mapping, label_scaler = encode_transform_scale(df, test_predict)
    data = scaled_df.tail(24)
    targets = data[['total travel delay']][23:]
    # print(f"[+] targets {targets}")
    data = data.drop(columns=['total travel delay'])

    test_predict = tf.keras.utils.timeseries_dataset_from_array(
        data=data, 
        targets=targets, 
        sequence_length=24,
        sequence_stride=1,
        sampling_rate=1,
        shuffle=False,
        batch_size=1)

    ypred = model.predict(test_predict)
    ypred = ypred.reshape(ypred.shape[0], 1)
    predicted = label_scaler.inverse_transform(ypred)
        
    st.write(predicted)


st.divider()
st.write("""
    ## Author
    - Frank Katada
    - MSCS I
    """
)