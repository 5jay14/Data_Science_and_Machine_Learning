# Conacatenation - It glues together the DF. DF dimension should match along the axis we are concatenating on.
# use pd.concat and pass the list of DF's tp concate together

import pandas as pd
import numpy as np

Mat1 = np.arange(25).reshape(5, 5)
Mat2 = np.arange(25, 50).reshape(5, 5)
Mat3 = np.arange(50, 75).reshape(5, 5)

DF1 = pd.DataFrame(Mat1, columns=['A', 'B', 'C', 'D', 'E'])
DF2 = pd.DataFrame(Mat2, columns=['A', 'B', 'C', 'D', 'E'])
DF3 = pd.DataFrame(Mat3, columns=['A', 'B', 'C', 'D', 'E'])

print(DF2)

print(pd.concat([DF1, DF2, DF3]))  # observe the indexes are sourced from the original df

# another example
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 4])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

print(pd.concat([df1, df2, df3]))
print(pd.concat([df1, df2, df3], axis=1))  # DF is stuck in column wise, missing values wil be filled with Nan

# Merging = Pandas allows to merge DF's together using a similar logics as merging sql tables together.
# Just like join in SQL, we can merge inner, outer, based on the key
# Key is a like primary key on which the merging takes places. Key can be more than 1
# How = 'inner', 'right','left','outer'

#Inner/natural = It will only return data that are there in both the DF
# Outer/outer full join = it will take everything from left, right
# left/left outer join = it will take everything from the left and similar from the right
# right/right outer join = opposite to left

left_df = pd.DataFrame({'Key': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']})

right_df = pd.DataFrame({'Key': ['A0', 'A1', 'A2', 'A3'],
                      'E': ['B4', 'B5', 'B6', 'B7'],
                      'F': ['C4', 'C5', 'C6', 'C7'],
                      'G': ['D4', 'D5', 'D6', 'D7']})

merge = pd.merge(left_df, right_df, how='inner', on='Key')
print(merge)

# Another example
left1 = pd.DataFrame({'Key1': ['A0', 'A1', 'A2', 'A3'],
                      'Key2': ['A0', 'A1', 'A0', 'A1'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

right1 = pd.DataFrame({'Key1': ['A0', 'A1', 'A2', 'A3'],
                       'Key2': ['A0', 'A3', 'A2', 'A4'],
                       'F': ['C4', 'C5', 'C6', 'C7'],
                       'G': ['D4', 'D5', 'D6', 'D7']})

merge1 = pd.merge(left1, right1, how='inner', on=['Key1', 'Key2'])
print(merge1)

# Joining= is a method of combining the columns of two potentially differently indexed DF into a single result df

left2 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['A0', 'A1', 'A0', 'A1']},
                      index=['D0', 'D1', 'D2', 'D3'])

right2 = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']},
                      index=['AF0', 'D1', 'FG2', 'D3'])

print(left2.join(right2))  # default will be inner
print(right2.join(left2, how='outer'))

# The main difference between join and merge in pandas is that join() is used to combine two DataFrames on the
# index but not on columns whereas merge() is primarily used to specify the columns you want to join on,
# this also supports joining on indexes and combination of index and columns
