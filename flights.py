import streamlit as st
import pandas as pd
import numpy as np

@st.cache(allow_output_mutation=True)

def load_pandas():
    pass

def load_model():
    pass

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

# Text
st.write("""
    # Hello Streamlit"""
)

# MAP
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

# Slider
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

# Checkbox
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

# Dropdown
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    key=1,
    label='Which number do you like best?',
    options=df['first column'])

'You selected: ', option

# Add a selectbox to the sidebar:
add_side_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

add_selectbox = st.selectbox(
    key=2,
    label='How would you like to be contacted?',
    options=('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_side_slider = st.sidebar.slider(
    key=3,
    label='Select a range of values',
    min_value=0.00, max_value=100.00, value=25.0
)

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