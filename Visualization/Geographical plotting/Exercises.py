import cufflinks as cf
from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs
import plotly.graph_objs as go
import pandas as pd

init_notebook_mode(connected=True)
cf.go_offline()

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\09"
                 r"-Geographical-Plotting\2014_World_Power_Consumption.csv")

data = dict(type='choropleth',
            colorscale='Viridis',
            reversescale=True,
            locations=df['Country'],
            locationmode="country names",
            z=df['Power Consumption KWH'],
            text=df['Text'],
            colorbar={'title': 'Power Consumption KWH'})

layout = dict(title='2012 Global Power Consumption',
              geo=dict(showframe=False,
                       projection={'type': 'natural earth'}))

choropleth = go.Figure(data=[data], layout=layout)
plot(choropleth, validate=False)

# **********************************************************

df1 = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\09"
                  r"-Geographical-Plotting\2012_Election_Data.csv")
print(df1.head())

data1 = dict(type='choropleth',
             locations=df1['State Abv'],
             locationmode='USA-states',
             text = df1['State'],
             colorscale='Portland',
             z=df1['Voting-Age Population (VAP)'],
             colorbar={'title': 'Voting-Age Population'})

layout1 = dict(geo={'scope': 'usa'})

choromap1 = go.Figure(data=[data1], layout=layout1)  # we are passing the dictionary inside a list,

plot(choromap1)
