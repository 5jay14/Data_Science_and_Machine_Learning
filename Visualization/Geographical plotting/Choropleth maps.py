"""
plotting on the information of global/nation scale
no need to memorize, use the template.
need to have two main objects/variables, data and layout and pass those into go.Figure and then map
need to maintain the order

visit plot.ly/python/reference/#choropleth
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cufflinks as cf
import plotly.graph_objs as go
from plotly.offline import (init_notebook_mode,iplot,iplot_mpl,plot_mpl,plot, download_plotlyjs)

init_notebook_mode(connected=True)
cf.go_offline()

# cast list into dictionary
# locations and location mode is available prebuilt
# Z is the datapoints/value for the locations
# type = type of geo plotting
# location is array/list of those actual state abbreviations. This is inbuilt which can go to county level
# text = array/list, text that shows when we hower that state in the map
data = dict(type='choropleth',
            locations=['AZ', 'CA', 'NY'],
            locationmode='USA-states',
            colorscale='Portland',
            text=['text1', 'text2', 'text3'],
            z=[1.0, 2.0, 3.0],
            colorbar={'title': 'colorbar title goes here'})

layout = dict(geo={'scope': 'usa'})

choromap = go.Figure(data=[data], layout=layout) # we are passing the dictionary inside a list,

plot(choromap)
