#############################################################################################################################
# IMPORT ALL MODULES
import streamlit as st    # python module for building web applications
import pandas as pd       # used for fetching and handling tabular data
import matplotlib.pyplot as plt        # a visualization tool
import seaborn as sns            # another visualization tool
import numpy as np        # for numerical analysis
import plotly.figure_factory as ff     # for interactive plots
import plost
#from PIL import Image
from itertools import combinations
#############################################################################################################################



#############################################################################################################################
# Helper Functions
def df_normalize(data_df, option = "columns"):
    """
    This function helps to normalize a dataframe that contains only numerical value.
    data_df: pandas.dataframe
    option: can be 'columns' or 'rows' to determine if normalizationj should be done across rows or columns
    
    output: normalized dataframe
    """
    
    norm_df = data_df.copy()
    
    if option == "rows":
        norm_df = norm_df.transpose()
        norm_df = (norm_df - norm_df.mean())/(norm_df.std())
        norm_df = norm_df.transpose()
        return norm_df
    
    norm_df = (norm_df - norm_df.mean())/(norm_df.std())
    return norm_df
#############################################################################################################################



#############################################################################################################################
# CONFIGURE WEB APPLICATION
# START
st.set_page_config(page_title=None, 
                   page_icon="chart_with_upwards_trend", 
                   layout="centered", 
                   initial_sidebar_state="auto", 
                   menu_items={
                                 'Get Help': 'https://www.extremelycoolapp.com/help',
                                 'Report a bug': "https://www.extremelycoolapp.com/bug",
                                 'About': "# This is a header. This is an *extremely* cool app!"
                               }
                  )
# LOAD STYLE SHEET FOR STYLING SOME WEB COMPONENTS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# SIDEBAR WIDTH CONFUGURATION
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 300px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Web page Title
st.markdown("""<h4 align="center">Maternal Health Risk Data Visualization</h4>""",unsafe_allow_html = True)
st.text(" ")
# END
#############################################################################################################################



#############################################################################################################################
# LOAD AND PREPROCESS DATASET
# START
data = pd.read_csv("Maternal Health Risk Data Set.csv")
# shuffle data
data.sample(frac=1)
# convert numerical columns to numbers
numeric_columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
group_labels = ['low risk', 'mid risk', 'high risk']
for column in numeric_columns:
    data[column] = data[column].astype(float)
    
units = {"Age" : "years", "SystolicBP" : "mmHg", "DiastolicBP" : "mmHg", 
         "BS" : "mmol/L", "HeartRate" : "BPM", 'BodyTemp' : "°F"}
# END
#############################################################################################################################



#############################################################################################################################
# PLOT DASHBOARD
# START
st.sidebar.title("Navigation")
option = st.sidebar.radio("Metrics",["All"]+group_labels)
if option == "All":
    usage_data = data
else:
    usage_data = data[data["RiskLevel"] == option]
# Row A
a1, a2, a3 = st.columns(3)
# Row B
b1, b2, b3 = st.columns(3)
st_columns = [a1,a2,a3,b1, b2, b3]
table_columns = usage_data.columns[:-1]
for i in range(len(st_columns)):
    mean = round(usage_data[table_columns[i]].mean())
    std = round(usage_data[table_columns[i]].std())
    unit = units[table_columns[i]]
    st_columns[i].metric(table_columns[i], f"{mean} {unit}", f"± {std} {unit}")
    
    
c1, c2 = st.columns([7,3])
with c2:
    st.text(" ")
    st.text(" ")
    st.text(" ")
    sub_data = {"RiskLevel": group_labels, "Count" : [0,0,0]}
    for item in data.RiskLevel:
        sub_data["Count"][group_labels.index(item)] += 1
    sub_data_df = pd.DataFrame(sub_data)
    plost.donut_chart(
        data=sub_data_df,
        theta='Count',
        color='RiskLevel')
    
with c1:
    pairs = [f"{pair[0]}, {pair[1]}" for pair in list(combinations(numeric_columns,2))]
    pair = st.sidebar.selectbox("Scatter Plots",pairs).split(", ")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)
    sns.scatterplot(data=data, x=pair[0], y=pair[1], hue="RiskLevel",style ="RiskLevel", legend='auto');
    st.pyplot(fig)
    
    
target_col = st.sidebar.radio("Histogram",numeric_columns)

d = sns.FacetGrid(data,hue="RiskLevel",height = 5, legend_out=False)
d.map(sns.distplot,target_col, kde=True)
d.add_legend();
d.fig.set_size_inches(15, 6)
st.pyplot(d.fig)
# END
#############################################################################################################################




#############################################################################################################################
# PLOT CLUSTER MAP
# START
# groups = {f"Group{(i+100)//100}" : (data.drop("RiskLevel", axis = 1).iloc[i:i+100,:], data["RiskLevel"][i:i+100]) for i in range(0,len(data),100)
#          }

# # groups["All"] = (data.drop("RiskLevel", axis = 1), data["RiskLevel"])
# group = st.sidebar.radio("Cluster Map",groups.keys())
# df = groups[group][0]
# sub_labels = groups[group][1]
# #Convert integers to floats 
# datafinal = df.astype(float) 
# #Perform log transformation using numpy package and show data description
# log = np.log(datafinal+1)
# norm_df = df_normalize(log, option = "columns");
# fig = sns.clustermap(norm_df, figsize = (20,25), yticklabels = sub_labels);
# st.pyplot(fig)
# END
#############################################################################################################################


      

#############################################################################################################################
#PLOT RELATIONSHIP MAP
# START
# Page setting
# st.graphviz_chart('''
#     digraph {
#         intr -> run
#         intr -> runbl
#         runbl -> run
#         run -> kernel
#         kernel -> zombie
#         kernel -> sleep
#         kernel -> runmem
#         sleep -> swap
#         swap -> runswap
#         runswap -> new
#         runswap -> runmem
#         new -> runmem
#         sleep -> runmem
#     }
# ''')
# END
#############################################################################################################################



#############################################################################################################################
# Page Info
st.sidebar.info(
"""
### Information
**Age:** Age in years when a woman is pregnant.  

**SystolicBP:** Upper value of Blood Pressure in mmHg, another significant attribute during pregnancy.  

**DiastolicBP:** Lower value of Blood Pressure in mmHg, another significant attribute during pregnancy.  

**BS:** Blood glucose levels is in terms of a molar concentration, mmol/L.  

**HeartRate:** A normal resting heart rate in beats per minute.  

**Risk Level:** Predicted Risk Intensity Level during pregnancy considering the previous attribute.  

**Dataset Source:** https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data  
"""
)
# END
#############################################################################################################################

