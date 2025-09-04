# In this project, we are not using the adaboost to classify the label but also give audience the guidance on which feature is
# importance

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# DF info
'''
Attribute Information:

cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
bruises?: bruises=t,no=f
odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
gill-attachment: attached=a,descending=d,free=f,notched=n
gill-spacing: close=c,crowded=w,distant=d
gill-size: broad=b,narrow=n
gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
stalk-shape: enlarging=e,tapering=t
stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
veil-type: partial=p,universal=u
veil-color: brown=n,orange=o,white=w,yellow=y
ring-number: none=n,one=o,two=t
ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
'''

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\mushrooms.csv")

# features_uni = df.describe().transpose().reset_index().sort_values('unique')
# sns.barplot(data=features_uni, x='index',y='unique')
# plt.xticks(rotation = 90)
# plt.show()

# sns.countplot(data = df, x='class')
# plt.show()


X = df.drop('class', axis=1)
X = pd.get_dummies(X, drop_first=True)
X = X.astype(int)
y = df['class']

print(X.info(), X.head(1))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report,accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = AdaBoostClassifier(n_estimators=1)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(y_pred)

print(model.feature_importances_)
print(model.feature_importances_.argmax()) # This will give the index of the feature that model is giving the importance
print(X.columns[22]) #Odor may be important now but as we add more stumps and adjusting the weights. feature imp will change

#  Model is performing with 88% accuracy with single feature and one stump
cr = classification_report(y_test,y_pred)
print(cr)

sns.countplot(data=df,x='odor', hue='class')
plt.show()
# We can notice for the odor = None, most of the mushrooms are edible although there are some poisonous mushorooms
# with no odor




error_rates = []
# 96 columns
for i in range (1,96):
    model = AdaBoostClassifier(n_estimators=i)
    model.fit(X_train,y_train)
    model_pred = model.predict(X_test)

    err = 1 - accuracy_score(y_test,model_pred)
    error_rates.append(err)


plt.plot(range(1,96),error_rates)
plt.show() # we can see the error stops around stump 18, there is no increase in performance after 20th stump
print(model) # will show the estimator being used
print(model.feature_importances_)

feats = pd.DataFrame(data=model.feature_importances_, index=X.columns, columns=['Importance'] )
imp_feats = feats[feats['Importance']>0].sort_values(by='Importance')
print(imp_feats)


sns.barplot(data=imp_feats,x=imp_feats.index,y='Importance')
plt.xticks(rotation =90)
plt.show()

















