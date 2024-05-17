import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data #(allow_output_mutation=True)
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

def load_model():
    pass

df: pd.DataFrame = load_pandas()
st.markdown("# TIP Finals: Model Deployment")

st.write("""
    ## Convolutional Neural Network Regression (Convo1D)
    Problem Statement: Predicting Arrival Flights Delay which is a Major problem specially with travelers with connecting flights
    """     
)
st.divider()
st.write("""
    ## Parameters
    """     
)

# st.sidebar.markdown("# Main page 🎈")

# Checkbox
if st.checkbox('Show dataframe'):
    st.dataframe(df)
    st.button("Rerun")

# Dropdown
option = st.selectbox(
    key=1,
    label='Origin Airport',
    options=df['origin'].unique())

'You selected: ', option

# Add a selectbox to the sidebar:
# add_side_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

add_selectbox = st.selectbox(
    key=2,
    label='How would you like to be contacted?',
    options=('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
# add_side_slider = st.sidebar.slider(
#     key=3,
#     label='Select a range of values',
#     min_value=0.00, max_value=100.00, value=25.0
# )

# Add a slider to the sidebar:
add_slider = st.slider(
    key=4,
    label='Select a range of values',
    min_value=0.00, max_value=100.00, value=25.0
)

left_column, right_column = st.columns(2)
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")


# State Plots
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("Choose a datapoint color")
color = st.color_picker("Color", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y", color=color)

st.divider()
st.write("""
    ## Author
    - Frank Katada
    - MSCS I
    """
)