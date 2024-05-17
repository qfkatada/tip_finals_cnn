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

# Month Day Year
departure_date = st.date_input(label="Departure Date",
                               help="Date of Departure"
                               )

# scheduled departure time
scheduled_departure_time = st.time_input( label="Scheduled Departure Time", help="Flight Scheduled Departure Time")

# departure time
departure_time = st.time_input(label="Time Departed", help="Actual time the plane departed")

# departure_delay = scheduled_departure_time - departure_time

# distance = origin + distination

# Dropdown
origin = st.selectbox(
    key=1,
    label='Airport Origin',
    options=df['origin'].unique(),
    )

'Airport Origin: ', origin

# Dropdown
airlines = st.selectbox(
    key=2,
    label='Airlines',
    options=df['name'].unique(),
    )

'Airlines: ', airlines

carrier = st.selectbox(
    key=3,
    label='Carrier Type',
    options=df['carrier'].unique(),
    )

'Carrier Type: ', airlines

# Dropdown
flight = st.selectbox(
        key=4,
        label='Airport Destination',
        options=df['flight'].unique(),
    )

'Flight: ', flight

st.write("""
    ### Destination
    """     
)

# Dropdown
destination = st.selectbox(
        key=5,
        label='Airport Destination',
        options=df['destination'].unique(),
    )

# Automap airport_destination, municipality, elavation
#
# airport_destination = st.selectbox(
#         key=3,
#         label='Airport Destination',
#         options=df['airport'].unique(),
#     )

# 

# scheduled departure time
scheduled_arrival_time = st.time_input(label="Arrival Time", help="The Scheduled Arrival time for the destination or connecting Airport")

# Default Values
described_values = df[['humidity','pressure', 'temperature', 'wind direction', 'wind speed', 'distance', 'eta duration']].describe()

eta_duration = st.number_input(
    label="Estimated Flight Dration", 
    min_value=described_values.loc['min','eta duration'], 
    help="ETA Flight Duration from the ticket")

st.write("""
    ## Weather Forecast
    """     
)

humidity = st.slider(
    key=6,
    label='Humidity',
    min_value=(described_values.loc['min', 'pressure'] - described_values.loc['std', 'pressure']), 
    max_value=(described_values.loc['max', 'pressure'] + described_values.loc['std', 'pressure']), 
    value=described_values.loc['mean', 'pressure']
)

pressure = st.slider(
    key=7,
    label='Pressure',
    min_value=(described_values.loc['min', 'pressure'] - described_values.loc['std', 'pressure']), 
    max_value=(described_values.loc['max', 'pressure'] + described_values.loc['std', 'pressure']), 
    value=described_values.loc['mean', 'pressure']
)

temperature = st.slider(
    key=8,
    label='Temperature',
    min_value=(described_values.loc['min', 'temperature'] - described_values.loc['std', 'temperature']), 
    max_value=(described_values.loc['max', 'temperature'] + described_values.loc['std', 'temperature']), 
    value=described_values.loc['mean', 'temperature']
)

wind_direction = st.slider(
    key=9,
    label='Wind Direction',
    min_value=(described_values.loc['min', 'wind direction'] - described_values.loc['std', 'wind direction']), 
    max_value=(described_values.loc['max', 'wind direction'] + described_values.loc['std', 'wind direction']), 
    value=described_values.loc['mean', 'wind direction']
)

wind_speed = st.slider(
    key=10,
    label='Wind Speed',
    min_value=(described_values.loc['min', 'wind speed'] - described_values.loc['std', 'wind speed']), 
    max_value=(described_values.loc['max', 'wind speed'] + described_values.loc['std', 'wind speed']), 
    value=described_values.loc['mean', 'wind speed']
)

weather_description = st.selectbox(
        key=11,
        label='Weather Description',
        options=df['weather description'].unique(),
        index=10
    )

st.write("""
    ## Possibility of missing a connecting flight due to arrival delay
    """     
)

predict = st.button("Predict", type="primary")
print(predict)
# if st.button("Predict"):
#     # Test Predict
#     pass

st.divider()
st.write("""
    ## Author
    - Frank Katada
    - MSCS I
    """
)