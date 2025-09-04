import pandas as pd
import cufflinks as cf
from plotly.offline import plot,init_notebook_mode,download_plotlyjs
import plotly.graph_objs as go

init_notebook_mode(connected=True)
cf.go_offline()
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\09"
                 r"-Geographical-Plotting\2014_World_GDP.csv")
print(df.info(), df.head())

data = dict(type= 'choropleth',
            locations = df['CODE'],
            z= df['GDP (BILLIONS)'],
            text = df['COUNTRY'],
            colorbar = {'title':'GDP in Billion USD'}
            )
layout = dict(title = '2014 Global GDP',
              geo =dict(showframe =False,
                        projection = {'type':'natural earth'}))
# few types
''' 'equirectangular' , 'mercator' , 'orthographic' , 'natural earth' , 'kavrayskiy7' , 'miller' , 'robinson' , 
'eckert4' , 'azimuthal equal area' , 'azimuthal equidistant' , 'conic equal area' , 'conic conformal' , 
'conic equidistant' , 'gnomonic' , 'stereographic' , 'mollweide' , 'hammer' '''
choropleth = go.Figure(data = [data],layout=layout)
plot(choropleth)