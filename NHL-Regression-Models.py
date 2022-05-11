#!/usr/bin/env python
# coding: utf-8

# In[1]:


# DATA 670 - Capstone, UMGC 2022
# Written by Joseph Coleman
# Last updated March 13, 2022
# The purpose of this code is to build decision tree, random forest, and multiple linear regression models
# to predict NHL fantasy points on Fanduel.
# The decision tree and random forest script was built with the help of Nick's Niche passion project. 


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import plot_tree


# In[3]:


g = pd.read_excel("C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\goalies_ml.xlsx")
w = pd.read_excel("C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\wings_ml.xlsx")
c = pd.read_excel("C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\centers_ml.xlsx")
d = pd.read_excel("C:\\Users\\jcole\\Documents\\DATA670\\machine_learning_players\\defensemen_ml.xlsx")


# In[4]:


gFeatureNames = ['Salary', 'Rest', 'Vegas.Odds.Win', 'Vegas.Odds.GA', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L40.SVPCT', 'S.SVPCT', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']
sFeatureNames = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']
labelName = ['Actual.FP']

gFeatures = g[['Salary', 'Rest', 'Vegas.Odds.Win', 'Vegas.Odds.GA', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L40.SVPCT', 'S.SVPCT', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
gLabel = g[['Actual.FP']]

wFeatures = w[['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
wLabel = w[['Actual.FP']]

cFeatures = c[['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
cLabel = c[['Actual.FP']]

dFeatures = d[['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
dLabel = d[['Actual.FP']]

gLabel = np.array(gLabel)
gFeatures = np.array(gFeatures)

wLabel = np.array(wLabel)
wFeatures = np.array(wFeatures)

cLabel = np.array(cLabel)
cFeatures = np.array(cFeatures)

dLabel = np.array(dLabel)
dFeatures = np.array(dFeatures)

gFeatures_s = gFeatures
wFeatures_s = wFeatures
cFeatures_s = cFeatures
dFeatures_s = dFeatures

gFeatures_s = StandardScaler().fit_transform(gFeatures_s)
wFeatures_s = StandardScaler().fit_transform(wFeatures_s)
cFeatures_s = StandardScaler().fit_transform(cFeatures_s)
dFeatures_s = StandardScaler().fit_transform(dFeatures_s)


# In[5]:


# decision tree - goalies
gtrain1, gtest1, gtrainLabels1, gtestLabels1 = train_test_split(gFeatures, gLabel, test_size = (0.4), random_state = 42)
g_1 = DecisionTreeRegressor(random_state = 42)
g_1.fit(gtrain1, gtrainLabels1)
g_pred1 = g_1.predict(gtest1)
g1 = pd.DataFrame(gtest1, columns = ['Salary', 'Rest', 'Vegas.Odds.Win', 'Vegas.Odds.GA', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L40.SVPCT', 'S.SVPCT', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
g1['Actual FP'] = gtestLabels1
g1['Predicted FP'] = g_pred1
g1['Error'] = abs(g1['Actual FP'] - g1['Predicted FP'])

# decision tree standardized - goalies
gtrain2, gtest2, gtrainLabels2, gtestLabels2 = train_test_split(gFeatures_s, gLabel, test_size = (0.4), random_state = 42)
g_2 = DecisionTreeRegressor(random_state = 42)
g_2.fit(gtrain2, gtrainLabels2)
g_pred2 = g_2.predict(gtest2)
g2 = pd.DataFrame(gtest2, columns = ['Salary', 'Rest', 'Vegas.Odds.Win', 'Vegas.Odds.GA', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L40.SVPCT', 'S.SVPCT', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
g2['Actual FP'] = gtestLabels2
g2['Predicted FP'] = g_pred2
g2['Error'] = abs(g2['Actual FP'] - g2['Predicted FP'])

# random forest - goalies
gtrain3, gtest3, gtrainLabels3, gtestLabels3 = train_test_split(gFeatures, gLabel, test_size = (0.4), random_state = 42)
g_3 = RandomForestRegressor(random_state = 42)
g_3.fit(gtrain3, gtrainLabels3)
g_pred3 = g_3.predict(gtest3)
g3 = pd.DataFrame(gtest3, columns = ['Salary', 'Rest', 'Vegas.Odds.Win', 'Vegas.Odds.GA', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L40.SVPCT', 'S.SVPCT', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
g3['Actual FP'] = gtestLabels3
g3['Predicted FP'] = g_pred3
g3['Error'] = abs(g3['Actual FP'] - g3['Predicted FP'])

# random forest standardized - goalies
gtrain4, gtest4, gtrainLabels4, gtestLabels4 = train_test_split(gFeatures_s, gLabel, test_size = (0.4), random_state = 42)
g_4 = RandomForestRegressor(random_state = 42)
g_4.fit(gtrain4, gtrainLabels4)
g_pred4 = g_4.predict(gtest4)
g4 = pd.DataFrame(gtest4, columns = ['Salary', 'Rest', 'Vegas.Odds.Win', 'Vegas.Odds.GA', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L40.SVPCT', 'S.SVPCT', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
g4['Actual FP'] = gtestLabels4
g4['Predicted FP'] = g_pred4
g4['Error'] = abs(g4['Actual FP'] - g4['Predicted FP'])

# decision tree - wings
wtrain1, wtest1, wtrainLabels1, wtestLabels1 = train_test_split(wFeatures, wLabel, test_size = (0.4), random_state = 42)
w_1 = DecisionTreeRegressor(random_state = 42)
w_1.fit(wtrain1, wtrainLabels1)
w_pred1 = w_1.predict(wtest1)
w1 = pd.DataFrame(wtest1, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
w1['Actual FP'] = wtestLabels1
w1['Predicted FP'] = w_pred1
w1['Error'] = abs(w1['Actual FP'] - w1['Predicted FP'])

# decision tree standardized - wings
wtrain2, wtest2, wtrainLabels2, wtestLabels2 = train_test_split(wFeatures_s, wLabel, test_size = (0.4), random_state = 42)
w_2 = DecisionTreeRegressor(random_state = 42)
w_2.fit(wtrain2, wtrainLabels2)
w_pred2 = w_2.predict(wtest2)
w2 = pd.DataFrame(wtest2, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
w2['Actual FP'] = wtestLabels2
w2['Predicted FP'] = w_pred2
w2['Error'] = abs(w2['Actual FP'] - w2['Predicted FP'])

# random forest - wings
wtrain3, wtest3, wtrainLabels3, wtestLabels3 = train_test_split(wFeatures, wLabel, test_size = (0.4), random_state = 42)
w_3 = RandomForestRegressor(random_state = 42)
w_3.fit(wtrain3, wtrainLabels3)
w_pred3 = w_3.predict(wtest3)
w3 = pd.DataFrame(wtest3, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
w3['Actual FP'] = wtestLabels3
w3['Predicted FP'] = w_pred3
w3['Error'] = abs(w3['Actual FP'] - w3['Predicted FP'])

# random forest standardized - wings
wtrain4, wtest4, wtrainLabels4, wtestLabels4 = train_test_split(wFeatures_s, wLabel, test_size = (0.4), random_state = 42)
w_4 = RandomForestRegressor(random_state = 42)
w_4.fit(wtrain4, wtrainLabels4)
w_pred4 = w_4.predict(wtest4)
w4 = pd.DataFrame(wtest4, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
w4['Actual FP'] = wtestLabels4
w4['Predicted FP'] = w_pred4
w4['Error'] = abs(w4['Actual FP'] - w4['Predicted FP'])

# decision tree - centers
ctrain1, ctest1, ctrainLabels1, ctestLabels1 = train_test_split(cFeatures, cLabel, test_size = (0.4), random_state = 42)
c_1 = DecisionTreeRegressor(random_state = 42)
c_1.fit(ctrain1, ctrainLabels1)
c_pred1 = c_1.predict(ctest1)
c1 = pd.DataFrame(ctest1, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
c1['Actual FP'] = ctestLabels1
c1['Predicted FP'] = c_pred1
c1['Error'] = abs(c1['Actual FP'] - c1['Predicted FP'])

# decision tree standardized - centers
ctrain2, ctest2, ctrainLabels2, ctestLabels2 = train_test_split(cFeatures_s, cLabel, test_size = (0.4), random_state = 42)
c_2 = DecisionTreeRegressor(random_state = 42)
c_2.fit(ctrain2, ctrainLabels2)
c_pred2 = c_2.predict(ctest2)
c2 = pd.DataFrame(ctest2, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
c2['Actual FP'] = ctestLabels2
c2['Predicted FP'] = c_pred2
c2['Error'] = abs(c2['Actual FP'] - c2['Predicted FP'])

# random forest - centers
ctrain3, ctest3, ctrainLabels3, ctestLabels3 = train_test_split(cFeatures, cLabel, test_size = (0.4), random_state = 42)
c_3 = RandomForestRegressor(random_state = 42)
c_3.fit(ctrain3, ctrainLabels3)
c_pred3 = c_3.predict(ctest3)
c3 = pd.DataFrame(ctest3, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
c3['Actual FP'] = ctestLabels3
c3['Predicted FP'] = c_pred3
c3['Error'] = abs(c3['Actual FP'] - c3['Predicted FP'])

# random forest standardized - centers
ctrain4, ctest4, ctrainLabels4, ctestLabels4 = train_test_split(cFeatures_s, cLabel, test_size = (0.4), random_state = 42)
c_4 = RandomForestRegressor(random_state = 42)
c_4.fit(ctrain4, ctrainLabels4)
c_pred4 = c_4.predict(ctest4)
c4 = pd.DataFrame(ctest4, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
c4['Actual FP'] = ctestLabels4
c4['Predicted FP'] = c_pred4
c4['Error'] = abs(c4['Actual FP'] - c4['Predicted FP'])

# decision tree - defensemen
dtrain1, dtest1, dtrainLabels1, dtestLabels1 = train_test_split(dFeatures, dLabel, test_size = (0.4), random_state = 100)
d_1 = DecisionTreeRegressor(random_state = 42)
d_1.fit(dtrain1, dtrainLabels1)
d_pred1 = d_1.predict(dtest1)
d1 = pd.DataFrame(dtest1, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
d1['Actual FP'] = dtestLabels1
d1['Predicted FP'] = d_pred1
d1['Error'] = abs(d1['Actual FP'] - d1['Predicted FP'])

# decision tree standardized - defensemen
dtrain2, dtest2, dtrainLabels2, dtestLabels2 = train_test_split(dFeatures_s, dLabel, test_size = (0.4), random_state = 42)
d_2 = DecisionTreeRegressor(random_state = 42)
d_2.fit(dtrain2, dtrainLabels2)
d_pred2 = d_2.predict(dtest2)
d2 = pd.DataFrame(dtest2, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
d2['Actual FP'] = dtestLabels2
d2['Predicted FP'] = d_pred2
d2['Error'] = abs(d2['Actual FP'] - d2['Predicted FP'])

# random forest - defensemen
dtrain3, dtest3, dtrainLabels3, dtestLabels3 = train_test_split(dFeatures, dLabel, test_size = (0.4), random_state = 42)
d_3 = RandomForestRegressor(random_state = 42)
d_3.fit(dtrain3, dtrainLabels3)
d_pred3 = d_3.predict(dtest3)
d3 = pd.DataFrame(dtest3, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
d3['Actual FP'] = dtestLabels3
d3['Predicted FP'] = d_pred3
d3['Error'] = abs(d3['Actual FP'] - d3['Predicted FP'])

# random forest standardized - defensemen
dtrain4, dtest4, dtrainLabels4, dtestLabels4 = train_test_split(dFeatures_s, dLabel, test_size = (0.4), random_state = 42)
d_4 = RandomForestRegressor(random_state = 42)
d_4.fit(dtrain4, dtrainLabels4)
d_pred4 = d_4.predict(dtest4)
d4 = pd.DataFrame(dtest4, columns = ['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP'])
d4['Actual FP'] = dtestLabels4
d4['Predicted FP'] = d_pred4
d4['Error'] = abs(d4['Actual FP'] - d4['Predicted FP'])


# In[6]:


# Multiple linear regression models.
gx = g[['Salary', 'Rest', 'Vegas.Odds.Win', 'Vegas.Odds.GA', 'L5.TOI', 'L40.TOI', 'S.TOI',
           'L40.SVPCT', 'S.SVPCT', 'L5.FP', 'L40.FP', 'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
gy = g[['Actual.FP']]

wx = w[['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI', 'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 
        'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
wy = w[['Actual.FP']]

cx = c[['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI', 'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 
        'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
cy = c[['Actual.FP']]

dx = d[['Salary', 'Rest', 'Line', 'PP', 'GS', 'L5.TOI', 'L40.TOI', 'S.TOI', 'L5.SOG', 'L40.SOG', 'S.SOG', 'L5.FP', 'L40.FP', 
        'S.FP', 'Floor.FP', 'Ceil.FP', 'Proj.FP']]
dy = d[['Actual.FP']]

gx_train, gx_test, gy_train, gy_test = train_test_split(gx, gy, test_size = 0.4, random_state = 42)
gmlr = LinearRegression()
gmlr.fit(gx_train, gy_train)
print("Intercept: ", gmlr.intercept_)
print("Coefficients:", gmlr.coef_)
gy_pred_mlr = gmlr.predict(gx_test)
g5 = pd.DataFrame()
g5['Actual FP'] = gy_test
g5['Predicted FP'] = gy_pred_mlr
g5['Error'] = abs(g5['Actual FP'] - g5['Predicted FP'])

wx_train, wx_test, wy_train, wy_test = train_test_split(wx, wy, test_size = 0.4, random_state = 42)
wmlr = LinearRegression()
wmlr.fit(wx_train, wy_train)
print("Intercept: ", wmlr.intercept_)
print("Coefficients:", wmlr.coef_)
wy_pred_mlr = wmlr.predict(wx_test)
w5 = pd.DataFrame()
w5['Actual FP'] = wy_test
w5['Predicted FP'] = wy_pred_mlr
w5['Error'] = abs(w5['Actual FP'] - w5['Predicted FP'])

cx_train, cx_test, cy_train, cy_test = train_test_split(cx, cy, test_size = 0.4, random_state = 42)
cmlr = LinearRegression()
cmlr.fit(cx_train, cy_train)
print("Intercept: ", cmlr.intercept_)
print("Coefficients:", cmlr.coef_)
cy_pred_mlr = cmlr.predict(cx_test)
c5 = pd.DataFrame()
c5['Actual FP'] = cy_test
c5['Predicted FP'] = cy_pred_mlr
c5['Error'] = abs(c5['Actual FP'] - c5['Predicted FP'])

dx_train, dx_test, dy_train, dy_test = train_test_split(dx, dy, test_size = 0.4, random_state = 42)
dmlr = LinearRegression()
dmlr.fit(dx_train, dy_train)
print("Intercept: ", dmlr.intercept_)
print("Coefficients:", dmlr.coef_)
dy_pred_mlr = dmlr.predict(dx_test)
d5 = pd.DataFrame()
d5['Actual FP'] = dy_test
d5['Predicted FP'] = dy_pred_mlr
d5['Error'] = abs(d5['Actual FP'] - d5['Predicted FP'])


# In[7]:


gmeanAbErr = metrics.mean_absolute_error(gy_test, gy_pred_mlr)
gmeanSqErr = metrics.mean_squared_error(gy_test, gy_pred_mlr)
grootMeanSqErr = np.sqrt(metrics.mean_squared_error(gy_test, gy_pred_mlr))
print('Adj R squared: {:.2f}'.format(gmlr.score(gx,gy)*100))
print('Mean Absolute Error:', gmeanAbErr)
print('Mean Square Error:', gmeanSqErr)
print('Root Mean Square Error:', grootMeanSqErr)

wmeanAbErr = metrics.mean_absolute_error(wy_test, wy_pred_mlr)
wmeanSqErr = metrics.mean_squared_error(wy_test, wy_pred_mlr)
wrootMeanSqErr = np.sqrt(metrics.mean_squared_error(wy_test, wy_pred_mlr))
print('R squared: {:.2f}'.format(wmlr.score(wx,wy)*100))
print('Mean Absolute Error:', wmeanAbErr)
print('Mean Square Error:', wmeanSqErr)
print('Root Mean Square Error:', wrootMeanSqErr)

cmeanAbErr = metrics.mean_absolute_error(cy_test, cy_pred_mlr)
cmeanSqErr = metrics.mean_squared_error(cy_test, cy_pred_mlr)
crootMeanSqErr = np.sqrt(metrics.mean_squared_error(cy_test, cy_pred_mlr))
print('R squared: {:.2f}'.format(cmlr.score(cx,cy)*100))
print('Mean Absolute Error:', cmeanAbErr)
print('Mean Square Error:', cmeanSqErr)
print('Root Mean Square Error:', crootMeanSqErr)

dmeanAbErr = metrics.mean_absolute_error(dy_test, dy_pred_mlr)
dmeanSqErr = metrics.mean_squared_error(dy_test, dy_pred_mlr)
drootMeanSqErr = np.sqrt(metrics.mean_squared_error(dy_test, dy_pred_mlr))
print('R squared: {:.2f}'.format(wmlr.score(dx,dy)*100))
print('Mean Absolute Error:', dmeanAbErr)
print('Mean Square Error:', dmeanSqErr)
print('Root Mean Square Error:', drootMeanSqErr)


# In[8]:


# how do our predictions do against the data source's predictions?

g1r = pd.DataFrame()
w1r = pd.DataFrame()
c1r = pd.DataFrame()
d1r = pd.DataFrame()
g2r = pd.DataFrame()
w2r = pd.DataFrame()
c2r = pd.DataFrame()
d2r = pd.DataFrame()
g3r = pd.DataFrame()
w3r = pd.DataFrame()
c3r = pd.DataFrame()
d3r = pd.DataFrame()
g4r = pd.DataFrame()
w4r = pd.DataFrame()
c4r = pd.DataFrame()
d4r = pd.DataFrame()
g5r = pd.DataFrame()
w5r = pd.DataFrame()
c5r = pd.DataFrame()
d5r = pd.DataFrame()
g6r = pd.DataFrame()
w6r = pd.DataFrame()
c6r = pd.DataFrame()
d6r = pd.DataFrame()

g6r['g6'] = abs(g['Actual.FP'] - g['Proj.FP'])
w6r['w6'] = abs(w['Actual.FP'] - w['Proj.FP'])
c6r['c6'] = abs(c['Actual.FP'] - c['Proj.FP'])
d6r['d6'] = abs(d['Actual.FP'] - d['Proj.FP'])

g1r['g1'] = g1['Error']
g2r['g2'] = g2['Error']
g3r['g3'] = g3['Error']
g4r['g4'] = g4['Error']
g5r['g5'] = g5['Error']

w1r['w1'] = w1['Error']
w2r['w2'] = w2['Error']
w3r['w3'] = w3['Error']
w4r['w4'] = w4['Error']
w5r['w5'] = w5['Error']

c1r['c1'] = c1['Error']
c2r['c2'] = c2['Error']
c3r['c3'] = c3['Error']
c4r['c4'] = c4['Error']
c5r['c5'] = c5['Error']

d1r['d1'] = d1['Error']
d2r['d2'] = d2['Error']
d3r['d3'] = d3['Error']
d4r['d4'] = d4['Error']
d5r['d5'] = d5['Error']


# In[9]:


# create df to compare results for each model
results = pd.concat([g1r, g2r, g3r, g4r, g5r, g6r, w1r, w2r, w3r, w4r, w5r, w6r, c1r, c2r, c3r, c4r, c5r, c6r, d1r, d2r, d3r, d4r, d5r, d6r], ignore_index = True)

pd.set_option('display.max_columns', None)

results.describe()


# In[10]:


dtresults = pd.concat([g1r, g2r, g6r, w1r, w2r, w6r, c1r, c2r, c6r, d1r, d2r, d6r])

dtresults.describe()


# In[11]:


rfresults = pd.concat([g3r, g4r, g6r, w3r, w4r, w6r, c3r, c4r, c6r, d3r, d4r, d6r])

rfresults.describe()


# In[12]:


mlrresults = pd.concat([g5r, g6r, w5r, w6r, c5r, c6r, d5r, d6r])

mlrresults.describe()

