import pandas as pd
import cufflinks as cf
import plotly.graph_objs as go
from plotly.offline import (init_notebook_mode,iplot,iplot_mpl,plot_mpl,plot, download_plotlyjs)

init_notebook_mode(connected=True)
cf.go_offline()

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\09"
                 r"-Geographical-Plotting\2011_US_AGRI_Exports.csv")
print(df.info(), df.head(1))
data = dict(type='choropleth',
            colorscale ='ylorrd',
            locations = df['code'],
            locationmode = 'USA-states',
            z = df['total exports'],
            text = df['text'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width =2)),#marker draws border line between each states
            colorbar = {'title':'Millions USD'})

layout = dict(title = '2011 US agriculture Exports by state',
              geo= dict(scope = 'usa', showlakes = True, lakecolor ='rgb(85,173,240)'))
choromap = go.Figure(data = [data], layout= layout)
plot(choromap)