import datetime
import decimal
from typing import Dict
import streamlit as st
import pandas as pd
import numpy as np

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
def scale(inputs: pd.DataFrame, labels: pd.DataFrame):
    pass

@st.cache
def load_model():
    model = load_model('./finals_streamlit/models/cnn_regression_base_lr0_0001epoch50drop02-F183')
    print(f"[+] {model.summary()}")
    return model

df: pd.DataFrame = load_pandas()
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

# st.sidebar.markdown("# Main page 🎈")

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
                               help="Date of Departure"
                               )

payload['month']= int(departure_date.month)
payload['day']= int(departure_date.day)

# scheduled departure time
scheduled_departure_time = st.time_input( label="Scheduled Departure Time", help="Flight Scheduled Departure Time")
payload['hour']= int(scheduled_departure_time.hour)
payload['scheduled departure time'] = int((scheduled_departure_time.hour*100) + (scheduled_departure_time.minute *10))

# departure time
departure_time = st.time_input(label="Time Departed", help="Actual time the plane departed")
payload['departure time'] = int((departure_time.hour*100) + (departure_time.minute *10))
payload['departure delay'] = payload['departure time'] - payload['scheduled departure time']

# distance = origin + distination

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

payload['municipality'] = municipality
payload['destination'] = destination
payload['elevation'] = elevation

# scheduled departure time
scheduled_arrival_time = st.time_input(label="Arrival Time", help="The Scheduled Arrival time for the destination or connecting Airport")
payload['scheduled arrival time'] = scheduled_arrival_time

# Default Values
described_values = df[['humidity','pressure', 'temperature', 'wind direction', 'wind speed', 'distance', 'eta duration']].describe()

eta_duration = st.number_input(
    label="Estimated Flight Dration", 
    # min_value=described_values.loc['min','eta duration'], 
    min_value=0.00, 
    max_value=described_values.loc['max', 'eta duration'],
    value=described_values.loc['mean', 'eta duration'],
    help="ETA Flight Duration from the ticket")

payload['eta duration'] = eta_duration

st.write("""
    ## Weather Forecast
    """     
)

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


if st.button("Predict",  type="primary"):
    _payload:Dict = {
        'month': departure_date.month,
        'day': departure_date.day,
        'hour': departure_time.hour,
        'origin': origin,
        'scheduled departure time': float(scheduled_departure_time.hour + (scheduled_departure_time.minute *100)),
        'departure time': 517.0,
        'departure delay': 2.0,
        'distance': 1400,
        'destination': 'IAH',
        'scheduled arrival time': 819,
        'eta duration': 304,
        'humidity': 81.0,
        'pressure': 1025.0,
        'temperature': 281.49,
        'weather description': 'broken clouds',
        'wind direction': 90.0,
        'wind speed': 4.0,
        'name': 'United Air Lines Inc.',
        'carrier': 'UA',
        'flight': 1545,
        'airport': 'George Bush Intercontinental Houston Airport',
        'municipality': 'Houston',
        'elevation': 97.0,
        # 'total travel delay': 13.0
    }

    _orig = list(_payload.keys())
    _proc = list(payload.keys())

    print("-"* 100)
    print(f"[+] original {len(_orig)}: {_orig}")
    print(f"[+] payload {len(_proc)}: {_proc}")
    print("-"* 100)

    for k in list(payload.keys()):
        if k in _orig:
           continue
        else:
            print(k)
    # print(f"[+] {payload}")

st.divider()
st.write("""
    ## Author
    - Frank Katada
    - MSCS I
    """
)