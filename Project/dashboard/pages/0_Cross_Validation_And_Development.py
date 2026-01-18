import streamlit as st
import pandas as pd
import plotly.express as px


st.markdown("<h1 style='text-align: center;'>Cross Validation & Model Development</h1>", unsafe_allow_html=True)

phase_colours = ['#FF6B6B', '#00CED1', '#8A2BE2']

def load_comparison_data():
    """
    Loads the summary comparison data (Baseline vs Single Test vs CV), from our 3 phases of development. 
    Manually retrieved from Model/notebook terminal. 
    """
    data = [
        # Trans
        {'Model': 'Transformer', 'Method': 'Baseline', 'RMSE': 40.71},
        {'Model': 'Transformer', 'Method': 'Single Test', 'RMSE': 0.74},
        {'Model': 'Transformer', 'Method': '5-Fold CV', 'RMSE': 1.44},
        # CNN
        {'Model': 'CNN (Neural Net)', 'Method': 'Baseline', 'RMSE': 1.71},
        {'Model': 'CNN (Neural Net)', 'Method': 'Single Test', 'RMSE': 1.44},
        {'Model': 'CNN (Neural Net)', 'Method': '5-Fold CV', 'RMSE': 1.02},
        # PF Transformer
        {'Model': 'PF Transformer', 'Method': 'Single Test', 'RMSE': 6.24},
        {'Model': 'PF Transformer', 'Method': '5-Fold CV', 'RMSE': 11.30},
        # PF CNN
        {'Model': 'PF CNN', 'Method': 'Single Test', 'RMSE': 22.24},
        {'Model': 'PF CNN', 'Method': '5-Fold CV', 'RMSE': 45.98},
    ]
    return pd.DataFrame(data)

def load_fold_data():
    folds = { ##Manually retrieved from notebook terminal
        'Transformer': [1.3639, 1.6124, 1.6009, 1.5483, 1.0965],
        'PF Transformer': [11.1119, 11.9866, 12.4137, 10.4978, 10.5122],
        'PF CNN': [45.0619, 43.9367, 48.3186, 45.6047, 46.9538],
    }
    rows = []
    
    for model, scores in folds.items():
        for i, score in enumerate(scores):
            rows.append({'Model': model, 'Fold': f'Fold {i+1}', 'RMSE': score})
    return pd.DataFrame(rows)

df_comparison = load_comparison_data()
df_folds = load_fold_data()



st.header("1. Core Models: Phase Development")
st.markdown("""
The two standalone models, the Tranformer and the CNN, followed different development trajectories over our 3-phase development/tuning.
""")

st.subheader("A. Transformer Development")

#Transformer
trans_base = 40.71
trans_cv = 1.44
transformer_improvement = ((trans_base - trans_cv) / trans_base) * 100
col_m1, col_m2 = st.columns([1, 3])
with col_m1:
    st.metric(label="Baseline RMSE", value=f"{trans_base}")
    st.metric(label="Final CV RMSE", value=f"{trans_cv}", delta=f"{transformer_improvement:.1f}% Improvement", delta_color="normal")
with col_m2:
    df_trans = df_comparison[df_comparison['Model'] == 'Transformer']
    fig_trans = px.bar(
        df_trans,
        x="Method",
        y="RMSE",
        color="Method",
        text_auto='.2f',
        color_discrete_sequence=phase_colours,
        title="Significant Error Reduction over phases"
    )
    fig_trans.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_trans, use_container_width=True)
st.markdown("---")


#CNN
st.subheader("B. CNN (Neural Net) Development")
cnn_base = 1.71
cnn_cv = 1.02
cnn_improvement = ((cnn_base - cnn_cv) / cnn_base) * 100
col_c1, col_c2 = st.columns([1, 3])
with col_c1:
    st.metric(label="Baseline RMSE", value=f"{cnn_base}")
    st.metric(label="Final CV RMSE", value=f"{cnn_cv}", delta=f"{cnn_improvement:.1f}% Improvement", delta_color="normal")
with col_c2:
    df_cnn = df_comparison[df_comparison['Model'] == 'CNN (Neural Net)']
    fig_cnn = px.bar(
        df_cnn,
        x="Method",
        y="RMSE",
        color="Method",
        text_auto='.2f',
        color_discrete_sequence=phase_colours,
        title="Exceptional Out-of-Box Performance with little room for improvement in absolute terms."
    )
    fig_cnn.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_cnn, use_container_width=True)
st.markdown("---")


## PF Models
st.header("2. Particle Filter Models")
st.caption("The two hybrid models, the Particle Filter augmented Transformer and the Particle Filter augmented CNN show no gains.")

pf_models = df_comparison[df_comparison['Model'].isin(['PF Transformer', 'PF CNN'])]
fig_pf = px.bar(
    pf_models,
    x="Model",
    y="RMSE",
    color="Method",
    barmode="group",
    text_auto='.2f',
    color_discrete_sequence=phase_colours[1:], 
    title="Particle Filter Hybrid Models"
)
fig_pf.update_layout(yaxis_title="RMSE")
st.plotly_chart(fig_pf, use_container_width=True)
st.markdown("---")

#Display df for CV
with st.expander("View Raw Fold Data for Select Models"):
    st.markdown("Detailed breakdown of RMSE across all 5 folds from one iteration.")
    pivot_folds = df_folds.pivot(index="Fold", columns="Model", values="RMSE")
    st.dataframe(pivot_folds)
