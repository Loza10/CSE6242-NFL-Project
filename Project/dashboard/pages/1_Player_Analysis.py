import streamlit as st
import pandas as pd
import plotly.express as px

st.markdown("<h1 style='text-align: center;'>Player Analysis</h1>", unsafe_allow_html=True)

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
select_models = all_sample_df[all_sample_df["Model"].isin(models_list)]

# Agg by Model and Player
calculation_df = (select_models.groupby(["Model", "nfl_id", "player_name"])["rmse"]
                  .mean().reset_index().rename(columns={"rmse": "mean_rmse"}))

# Func to create sorted squared heatmap
def create_grid_heatmap(df, model_name, items_per_row=8):
    """
    Creates a sorted grid of squares/waffle type heatmap for players. Error by player ID.
    """
    
    df_subset = df[df['Model'] == model_name].copy()
    # Sort by RMSE 
    df_subset = df_subset.sort_values(by="mean_rmse", ascending=False).reset_index(drop=True)

    df_subset['row'] = df_subset.index // items_per_row
    df_subset['col'] = df_subset.index % items_per_row
    # Create Label (Nfl_id/Name for hover)
    df_subset['label'] = df_subset['nfl_id'].astype(str)

    waffle_fig = px.scatter(
        df_subset,
        x='col',
        y='row',
        color='mean_rmse',
        symbol_sequence=['square'], # Forces squares, circle otherwise
        text='label',               # Puts NFL ID (which is the player ID) inside the square
        hover_data=['player_name', 'mean_rmse'],
        color_continuous_scale='RdBu_r', # Red = High Error
        size_max=40,
        title=f"{model_name}: Player ID RMSE Heatmap"
    )

    waffle_fig.update_traces(
        marker=dict(size=45, line=dict(width=1, color='DarkSlateGrey')),
        textfont=dict(color='white', size=10) # Ensure text is readable
    )
    waffle_fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(visible=False, showgrid=False), # Hide axes
        yaxis=dict(visible=False, showgrid=False, autorange="reversed"), # Top-down
        height=max(600, len(df_subset) // items_per_row * 60), # Dynamic height
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return waffle_fig

st.header("As we can see, the CNN model has noticeably lower average RMSE for most players.")

first_tab, second_tab = st.tabs(["Transformer Heatmap", "CNN Heatmap"])
with first_tab:
    fig_trans = create_grid_heatmap(calculation_df, "Transformer")
    st.plotly_chart(fig_trans, use_container_width=True)
with second_tab:
    fig_cnn = create_grid_heatmap(calculation_df, "CNN (Neural Net)")
    st.plotly_chart(fig_cnn, use_container_width=True)

st.markdown("---") # Line separator

st.header("Top & Bottom Performance Analysis")
st.caption("On average, which players did each model perform the best and worst with?")

c1, c2 = st.columns(2)
with c1:
    select_model = st.selectbox("Select Model to Analyze:", models_list)
with c2:
    top_or_bottom_n = st.selectbox("Show Top/Bottom N Players:", [5, 10, 15, 20])


# Filter for the selected model
model_subset = calculation_df[calculation_df["Model"] == select_model]

# Best = Lowest RMSE Asc
best_players = model_subset.sort_values("mean_rmse", ascending=True).head(top_or_bottom_n)

# Worst = Highest RMSE Desc
worst_players = model_subset.sort_values("mean_rmse", ascending=False).head(top_or_bottom_n)

#Side by side
col_best, col_worst = st.columns(2)

# Two graphs, equal, side by side. Left Best, Right worst
with col_best:
    st.subheader(f"Top {top_or_bottom_n} Best Players (Lowest Error)")
    fig_best = px.bar(
        best_players,
        x="mean_rmse",
        y="player_name",
        orientation='h',      
        text_auto='.3f',
        color="mean_rmse",
        color_continuous_scale="Teal_r", #For top picks
        title=f"Best Performers ({select_model})"
    )
    fig_best.update_layout(yaxis={'categoryorder':'total descending'}, showlegend=False)
    st.plotly_chart(fig_best, use_container_width=True)

# Right Worst
with col_worst:
    st.subheader(f"Worst {top_or_bottom_n} Players (Highest Error)")
    fig_worst = px.bar(
        worst_players,
        x="mean_rmse",
        y="player_name",
        orientation='h',
        text_auto='.3f',
        color="mean_rmse",
        color_continuous_scale="Reds", # For worse pickks
        title=f"Worst Performers ({select_model})"
    )
    fig_worst.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False) #Sorts squres
    st.plotly_chart(fig_worst, use_container_width=True)
