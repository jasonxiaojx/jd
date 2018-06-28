
# coding: utf-8

# In[1]:


"""
Avaliable resources:
#sku_attr.csv --- side characteristics
#sku_cost.csv
sku_discount_testing_2018MarApr
#sku_info
sku_prom
sku_prom_testing_2018MarApr
sku_sales
#sku_stock

# = insignificant to predictions

"""
import numpy as np
import pandas as pd


# Proprocessing DataTables into DataFrames PD:
sku_sales = pd.read_csv('sku_sales.csv')
sku_sales['date'] = pd.to_datetime(sku_sales['date'])
sku_sales = sku_sales.set_index('item_sku_id')
sku_std = pd.read_csv('par_RDC_total_sale.csv')
sku_std = sku_std.set_index('item_sku_id')
sku_sales['discount'].fillna(0, inplace = True)
sku_sales['vendibility'].fillna(0, inplace = True)
sku_prom = pd.read_csv('sku_prom.csv')
sku_prom['date'] = pd.to_datetime(sku_prom['date'])
sku_prom = sku_prom.set_index('item_sku_id')



# In[441]:

def slicerp(t0, a7):
    cut = sku_prom[sku_prom['date']>= t0]
    cut2 = cut[cut['date'] <= a7]
    return cut2

def prom(t0, a7):
    sku_ids = [i for i in range(1, 1001)]
    shell = pd.DataFrame(index = sku_ids, columns = [1,4,6,10]).fillna(0)
    prom_data = pd.get_dummies(slicerp(t0, a7)['promotion_type'])
    non_repeated = list(set(prom_data.index))
    for i in non_repeated:
        a = prom_data[prom_data.index == i][1].sum()
        b = prom_data[prom_data.index == i][4].sum()
        c = prom_data[prom_data.index == i][6].sum()
        d = prom_data[prom_data.index == i][10].sum()
        shell.loc[i, 1] = a
        shell.loc[i, 4] = b
        shell.loc[i, 6] = c
        shell.loc[i, 10] = d
    return shell



# In[477]:


def slicerx(t30, t0, dc_id):
    time_slice = sku_sales[sku_sales['date'] >= t30]
    time_slice2 = time_slice[time_slice['date'] <= t0]
    dc_slice = time_slice2[time_slice2['dc_id'] == dc_id].sort_values('date')
    return dc_slice

def preprocess(t0, t7, t14, t30, a7, dc_id):
    p_7_raw = slicerx(t7, t0, dc_id)
    p_7 = p_7_raw.groupby(p_7_raw.index)['quantity'].sum()
    p_14_raw = slicerx(t14, t0, dc_id)
    p_14 = p_14_raw.groupby(p_14_raw.index)['quantity'].sum()
    p_30_raw = slicerx(t30, t0, dc_id)
    p_30 = p_30_raw.groupby(p_30_raw.index)['quantity'].sum()
    a_7_raw = slicerx(t0, a7, dc_id)
    discount = a_7_raw.groupby(a_7_raw.index)['discount'].mean()
    joined = pd.concat([p_7, p_14, p_30, discount], axis=1)
    promotions = prom(t0, a7)
    join_prom = pd.concat([joined, promotions], axis = 1)
    join_prom.columns = ['p_7', 'p_14','p_30','discount', '1', '4', '6', '10']
    a_7 = a_7_raw.groupby(a_7_raw.index)['quantity'].sum()
    join_prom = join_prom.fillna(0)
    a_7 = a_7.fillna(0)
    return [join_prom,a_7]




#Machine Learning Regressors
from sklearn.ensemble import GradientBoostingRegressor #Lowest MAPE with 0.75 atm.
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR


# In[487]:


#Training Process (doesnt account for size difference possibilities in XY Training Data)
#After prediction, the prediction is modified via upperbound, capped by r times the std + mean.
#Returns df[predicted, true].
def upper_bound(predicted, sku_std, r):
    def rsd(r, mu, sd):
        return mu + r * sd
    bound = sku_std.apply(lambda x: rsd(r, sku_std.sku_avg, sku_std.sku_std), axis = 0)
    bound = bound['sku_avg']
    for i in predicted.index:
        if predicted[predicted.index == i].values[0] > bound[bound.index == i].values[0]:
            predicted[predicted.index == i] = bound[bound.index == i]
    return predicted

def predict(trainer, testing):
    rgs = GradientBoostingRegressor()
    rgs.fit(trainer[0], trainer[1])
    predicted = rgs.predict(testing[0])
    predicted_sku = testing[0].index
    true = testing[1]
    drop = []
    for i in range(len(predicted)):
        if predicted[i] < 0:
            predicted[i] = 0
    for i in true.index:
        if i not in predicted_sku:
            drop.append(i)
    true = true.drop(drop)
    transformed = pd.Series(data = predicted, index = predicted_sku)
    #applying bounds
    transformed = upper_bound(transformed, sku_std, 4)
    output = pd.concat([true, transformed], axis = 1)
    output.columns = ['true', 'predicted']
    return output/7

def MAPE(result):
    true = result['true'].values
    predicted = result['predicted'].values
    percentage_errors = []
    for i in range(len(true)):
        if true[i] > 0:
            percentage_errors.append(abs((true[i]-predicted[i])/true[i]))
    print(sum(percentage_errors)/len(percentage_errors))



total_predictions = []
for i in range(6):
    trainer = preprocess('20180201', '20180125', '20180118', '20180101', '20180208', i)
    testing = preprocess('20180101', '20171225', '20171218', '20171201', '20180108', i)
    total_predictions.append(predict(trainer, testing)['predicted'])
lol = pd.concat(total_predictions, axis = 1)
lol.columns = [0, 1, 2, 3, 4, 5]



def add(a, b, c, d, e, f):
    return a + b + c + d + e + f
lol['sum'] = lol.apply(lambda x:add(lol[0], lol[1], lol[2], lol[3], lol[4], lol[5]))[0]
#FILE OUTPUT
lol.to_csv('aggregate.csv')


stacked = pd.DataFrame()
stacked['prediction'] = lol[1].values
stacked['dc_id'] = 1
stacked.index = lol.index
for i in range(2,6):
    temp = pd.DataFrame()
    temp['prediction'] = lol[i].values
    temp['dc_id'] = i
    temp.index = lol.index
    stacked = stacked.append(temp)
    print(temp)

#FILE OUTPUT
stacked.to_csv('stacked.csv')
