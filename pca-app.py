import streamlit as st
import pandas as pd
import plotly.express as px
from function import pca_maker
from sklearn.decomposition import PCA
import numpy as np
import time
pip install --use-pep517
pip install ez_setup

st.set_page_config(layout="wide")
scatter_column, settings_column = st.beta_columns((4, 1))

scatter_column.title("PCA for Multi-Dimensional Analysis demo")

scatter_column.write("Here we perform Dimentionality reduction using PCA where we reduce a Multi-Dimensional dataset to 2 dimentions for the purpose of visualization.")

scatter_column.write("The first 2 principal components (PCA 1 and PCA 2) contain the direction with the maximum variance and hence provide the most useful information when visualized on a 2-D scale.")
scatter_column.write("Here iris dataset is selected by default. You can upload your own dataset under the settings column on the right.")
scatter_column.write("For iris dataset 4 columns namely sepal_length , sepal_width , petal_length , petal_width (as seen in raw data section) is reduced to its principle components and the top two (PCA 1 and PCA 2) is plotted on the 2-D scale. Thus reducing it from 4-D to 2-D for visualization.")
settings_column.title("Settings")

uploaded_file = settings_column.file_uploader("Choose a .csv File - if no file is choosen in 5 seconds(reload to restart timer), by default iris dataset is used")

def perform_pca(data_import):
	pca , scaled_values , pca_data, cat_cols, pca_cols , num_data = pca_maker(data_import)
	categorical_variable = settings_column.selectbox("Variable Select", options = cat_cols)
	categorical_variable_2 = settings_column.selectbox("Second Variable Select", options = cat_cols)
	pca_1 = settings_column.selectbox("First Principle Component (x-axis)", options=pca_cols, index=0)
	pca_cols.remove(pca_1)
	pca_2 = settings_column.selectbox("Second Principle Component(y-axis)", options=pca_cols)


	scatter_column.plotly_chart(px.scatter(data_frame=pca_data, x=pca_1, y=pca_2, color=categorical_variable, template="simple_white", height=800, hover_data = [categorical_variable_2]), use_container_width=True)

	st.subheader('Raw data')
	st.write(data_import)

	st.subheader('Transformed data')
	st.write(scaled_values)

	st.subheader('PCA data')
	st.write(pca_data)
	st.subheader('Explore loadings')

	loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

	loadings_df = pd.DataFrame(loadings)
	loadings_df = pd.concat([loadings_df, 
                         pd.Series(num_data, name='var')], 
                         axis=1)

	component = st.selectbox('Select PCA component:', loadings_df.columns[0:4])

	bar_chart = px.bar(loadings_df[['var', component]].sort_values(component), 
                   y='var', 
                   x=component, 
                   orientation='h',
                   range_x=[-1,1])


	st.write(bar_chart)

	st.write("Here we see the impact of different variables for each principal component(here PCA component 0 represents PCA 1)")
	
	
time.sleep(5)



if uploaded_file is not None:
	data_import = pd.read_csv(uploaded_file)

	perform_pca(data_import)
	


elif uploaded_file is None:
	iris = px.data.iris().drop('species_id', axis=1)
	perform_pca(iris)
