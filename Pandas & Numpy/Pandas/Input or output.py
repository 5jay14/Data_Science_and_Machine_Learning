# Dealing with CSV, Excel, HTML and SQL
# Pandas can read from multiple sources
import pandas as pd
import numpy as np
import io as iio
from sqlalchemy import create_engine

# #Note : \ is a escape character, change \ to / in the path or add 'r' before the path string literal
# a= pd.read_csv("C:/Users/vijay/Desktop/example.csv")
# print(a)
# print(a.info()) #this gives info about how many entries there are
# # write to a csv file, from a df, we can use .to
# df = a.to_csv('Output',index=False)

# working with excel, pandas can only work with data and it does not import macros, formulas or images
# can also specify the sheet name
# AtoM= pd.read_excel(r"C:\Users\vijay\Desktop\Aunty to Mom.xlsx")
# print(AtoM)
# AtoM.to_excel("C:/Users/vijay/Desktop/Aunty to Mom2.xlsx",sheet_name='new1')

url = 'https://en.wikipedia.org/wiki/List_of_UFC_champions'
#reading HTML
b= pd.read_html(url)


print(type(b)) # it is a list as pandas will reference to list class in html
print(b[0])
b1 = b[0].to_excel("C:/Users/vijay/Desktop/ufc.xlsx",sheet_name='new1')

engine = create_engine('sqlite:///:memory:') #This is constant, this creates a sql engine
ss = b[0].to_sql('My_table',con=engine) #name and engine
print(ss)