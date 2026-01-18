import streamlit as st
import pandas as pd
import plotly.express as px

st.markdown("<h1 style='text-align: center;'>Position Analysis</h1>", unsafe_allow_html=True)

# 2. Load Data
@st.cache_data
def load_data():
    try:
        df_all = pd.read_csv("data/all_results.csv")
    except FileNotFoundError:
        st.error("File 'data/all_results.csv' not found. Or you're launching the streamlit app from the wrong directory. See README for troubleshooting.")
        return None
    names_dict_b = {
        'neural_net': 'CNN (Neural Net)',
        'pf_neural_net': 'PF CNN',
        'pf_none': 'PF Baseline',
        'pf_transformer': 'PF Transformer',
        'transformer': 'Transformer'
    }
    
    df_all['model_type'] = df_all['model_type'].replace(names_dict_b)
    df_all = df_all.rename(columns={'model_type': 'Model'})
    return df_all

all_sample_df = load_data()


models_list = ['Transformer', "CNN (Neural Net)"]
radar_data = all_sample_df[all_sample_df["Model"].isin(models_list)]


# Agg to get metric
radar_df = radar_data.groupby(['Model', 'player_position'])['rmse'].mean().reset_index()


st.subheader("Radar Chart: RMSE by Field Position")
st.caption("Comparing the shape of error. Closer to center significies better performance. Both scales are independent. Although the CNN had better overall performance, it still performed better on certain positions than others relative to itself.")

col1, col2 = st.columns(2)

#Left radar: Transformer
with col1:
    st.markdown("#### Transformer")
    df_trans = radar_df[radar_df['Model'] == 'Transformer']
    
    fig_trans = px.line_polar(
        df_trans,
        r='rmse',
        theta='player_position',
        line_close=True,
        markers=True,
        title=""
    )
    fig_trans.update_traces(fill='toself', line_color='#636EFA')
    fig_trans.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig_trans, use_container_width=True)

# Right CNN
with col2:
    st.markdown("#### CNN (Neural Net)")
    df_cnn = radar_df[radar_df['Model'] == 'CNN (Neural Net)']
    
    fig_cnn = px.line_polar(
        df_cnn,
        r='rmse',
        theta='player_position',
        line_close=True,
        markers=True,
        title=""
    )
    # Red Fill
    fig_cnn.update_traces(fill='toself', line_color='#EF553B')
    fig_cnn.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig_cnn, use_container_width=True)


# Line chart under the side by side radar
st.markdown("---")
st.header("Line Chart")
st.caption("Linear comparison of RMSE acros all positions. CNN is the clear winner across all different positions")

fig_line = px.line(
    radar_df, 
    x="player_position", 
    y="rmse", 
    color="Model",
    markers=True,
    title="Average RMSE by Position: Transformer vs CNN",
    # SaMe colours as before
    color_discrete_map={
        "Transformer": "#636EFA",
        "CNN (Neural Net)": "#EF553B"
    }
)

fig_line.update_layout(xaxis_title="Position", yaxis_title="Mean RMSE")
st.plotly_chart(fig_line, use_container_width=True)
