import streamlit as st
import pandas as pd
from pathlib import Path
from nfl_visualizer import create_play_animation 

##Amir's Code/Script, slightly adapted to fit into dashboard
st.title("Interactive Play Animation")
st.caption("Select a Play ID to visualize player movement and ball trajectory.")


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at: {file_path}")
        return None

input_file = 'data/input_2023_w01.csv'
df_all = load_data(input_file)

#Ids
play_ids = sorted(df_all['play_id'].unique())

selected_play_id = st.selectbox(
    "Play ID", 
    play_ids,
    index=play_ids.index(101) if 101 in play_ids else 0
)


#Drop down filter
play_df = df_all[df_all['play_id'] == selected_play_id]

fig = create_play_animation(play_df, selected_play_id, show_ball=True)

#Plot it
st.plotly_chart(fig, use_container_width=True)
