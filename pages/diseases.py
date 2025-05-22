# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 01:17:43 2025
The file if for the blood markers information page
@author: zhper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

# Load the data
df1 = pd.read_csv('Data/Blood_samples_dataset_balanced_2.csv')
df2 = pd.read_csv('Data/blood_samples_dataset_test.csv')
df = pd.concat([df1, df2], ignore_index=True)

df = df.replace('Thalasse', 'Thalassemia').replace('Heart Di', 'Heart Disease')
pd.set_option("display.max_columns", None)

diseases = df["Disease"].unique()
markers = df.columns[:-1]
# Getting correlation between markers
# Dropping categorical data column
def correlation_graph():
    drop_disease = df.drop('Disease', axis=1)
    corr = drop_disease.corr()
    matrix = np.triu(np.ones(corr.shape, dtype=bool))
    corr = corr.mask(matrix)
    fig = px.imshow(corr, color_continuous_scale="Blues", width=800, height=800)
    return fig


def whisker_plot(ailment):
    melted = df.melt(id_vars="Disease", var_name="Marker_Stat", value_name="Value")
    df_ailment = melted[melted.Disease == ailment]
    fig = px.box(df_ailment, x="Marker_Stat", y="Value", color="Marker_Stat")
    fig.update_xaxes(tickangle=45)
    return fig

def violins(markers):
    df_melt = df.melt(id_vars='Disease', value_vars=markers, var_name='Marker', value_name='Marker_Value')
    fig = px.violin(df_melt, x="Disease", y='Marker_Value', color='Marker')
    return fig

st.markdown("<h3 style='font-size:24px;'>What markers would you like to view?</h3>", unsafe_allow_html=True)
option = st.selectbox('', np.sort(diseases))

st.subheader(f"Blood Markers for {option}")
st.plotly_chart(whisker_plot(option))

st.markdown("<h3 style='font-size:24px;'>🦠 How do markers compare?</h3>", unsafe_allow_html=True)

select_markers = [marker for marker in markers if st.checkbox(marker)]
if st.button('Compare'):
    st.plotly_chart(violins(select_markers))

st.subheader('Correlational heatmap')
st.markdown("<h4 style='font-size:24px;'>The graph below demostrates the correlations between different markers. The darker the color, the more positively correlated they are (In the dataset, it means that often, an increase in one value happens at the same time as the increase in another value).</h4>", unsafe_allow_html=True)

st.plotly_chart(correlation_graph(), use_container_width=True)