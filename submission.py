#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
sample submission for 2nd round
'''

import pandas as pd
import numpy as np
# import all modules been used

class UserPolicy:
    def __init__(self, initial_inventory, sku_cost):
        self.inv = [initial_inventory]
        self.costs = sku_cost
        self.extra_shipping_cost_per_unit = 0.01
        self.fixed_replenish_cost = 0.01
        self.sku_limit = np.asarray([200, 200, 200, 200, 200])
        self.capacity_limit = np.asarray([3200, 1600, 1200, 3600, 1600])
        self.abandon_rate =np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])


    def daily_decision(self, t):
        '''
        daily decision of inventory allocation
        input values:
            t, decision date
        return values:
            inventory decision, 2-D numpy array, shape (6,1000), type integer
        '''
        # Importing ML Modules from SKLEARN:
        from sklearn.ensemble import GradientBoostingRegressor #Lowest MAPE with 0.75 atm.
        #from sklearn.linear_model import BayesianRidge
        #from sklearn.tree import DecisionTreeRegressor
        #from sklearn.kernel_ridge import KernelRidge
        #from sklearn.svm import SVR

        # Importing and Formatting Data:
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
        sku_prom_sub = pd.read_csv('sku_prom_testing_2018MarApr.csv')
        sku_prom_sub['date'] = pd.to_datetime(sku_prom_sub['date'])
        sku_prom_sub = sku_prom_sub.set_index('item_sku_id')
        sku_discount_sub = pd.read_csv('sku_discount_testing_2018MarApr.csv').fillna(0)
        sku_discount_sub['date'] = pd.to_datetime(sku_discount_sub['date'])
        # Your algorithms here
        # Replenishment --- Data Organization:
        def slicerp(t0, a7):
            cut = sku_prom[sku_prom['date']>= t0]
            cut2 = cut[cut['date'] <= a7]
            return cut2

        def slicerp_sub(t0, a7):
            cut = sku_prom_sub[sku_prom_sub['date']>= t0]
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

        def prom_sub(t0, t7):
            sku_ids = [i for i in range(1, 1001)]
            shell = pd.DataFrame(index = sku_ids, columns = [1,4,6,10]).fillna(0)
            prom_data = pd.get_dummies(slicerp_sub(t0, a7)['promotion_type'])
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

        def slicerd(t0, a7, dc_id):
            time_slice = sku_discount_sub[sku_discount_sub['date'] <= a7]
            time_slice2 = time_slice[time_slice['date'] >= t0]
            dc_slice = time_slice2[time_slice2['dc_id'] == dc_id].sort_values('date')
            return dc_slice

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

        def preprocess_sub(t0, t7, t14, t30, a7, dc_id):
            p_7_raw = slicerx(t7, t0, dc_id)
            p_7 = p_7_raw.groupby(p_7_raw.index)['quantity'].sum()
            p_14_raw = slicerx(t14, t0, dc_id)
            p_14 = p_14_raw.groupby(p_14_raw.index)['quantity'].sum()
            p_30_raw = slicerx(t30, t0, dc_id)
            p_30 = p_30_raw.groupby(p_30_raw.index)['quantity'].sum()
            d_7_raw = slicerd(t0, a7, dc_id)
            discount = d_7_raw.groupby(d_7_raw.index)['discount'].mean()
            joined = pd.concat([p_7, p_14, p_30, discount], axis=1)
            promotions = prom_sub(t0, a7)
            join_prom = pd.concat([joined, promotions], axis = 1)
            join_prom.columns = ['p_7', 'p_14','p_30','discount', '1', '4', '6', '10']
            join_prom = join_prom.fillna(0)
            return join_prom

        # Prediction Alterations and Training:
        def upper_bound(predicted, sku_std, r):
            def rsd(r, mu, sd):
                return mu + r * sd
            bound = sku_std.apply(lambda x: rsd(r, sku_std.sku_avg, sku_std.sku_std), axis = 0)
            bound = bound['sku_avg']
            for i in predicted.index:
                if predicted[predicted.index == i].values[0] > bound[bound.index == i].values[0]:
                    predicted[predicted.index == i] = bound[bound.index == i]
            return predicted

        def predict(trainer, testing, sub = False):
            if sub:
                rgs = GradientBoostingRegressor()
                rgs.fit(trainer[0], trainer[1])
                predicted = rgs.predict(testing)
                predicted_sku = testing[0].index
                transformed = pd.Series(data = predicted, index = predicted_sku)
                #applying bounds
                transformed = upper_bound(transformed, sku_std, 4)
                return transformed
            else:
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

        def add(a, b, c, d, e, f):
            return a + b + c + d + e + f
        def replenishment_ary():
            total_predictions = []
            for i in range(6):
                trainer = preprocess('20180201', '20180125', '20180118', '20180101', '20180208', i)
                testing = preprocess_sub('20180101', '20171225', '20171218', '20171201', '20180108', i)
                total_predictions.append(predict(trainer, testing, sub = True))
            lol = pd.concat(total_predictions, axis = 1)
            lol.columns = [0, 1, 2, 3, 4, 5]
            lol['sum'] = lol.apply(lambda x:add(lol[0], lol[1], lol[2], lol[3], lol[4], lol[5]))[0]
            stacked = pd.DataFrame()
            stacked['sku_avg'] = lol[1].values
            stacked['dc_id'] = 1
            stacked['sku_std'] = sku_std['sku_std'].values
            stacked.index = lol.index
            for i in range(2,6):
                temp = pd.DataFrame()
                temp['sku_avg'] = lol[i].values
                temp['dc_id'] = i
                temp['sku_std'] = sku_std['sku_std'].values
                temp.index = lol.index
                stacked = stacked.append(temp)
                print(temp)

            replenishment_rdc = pd.concat([lol['sum'],sku_std['sku_std']], axis = 1)
            replenishment_rdc.columns = ['sku_avg', 'sku_std']
            replenishment_rdc['item_sku_id'] = replenishment_rdc.index
            replenishment_fdc = stacked
            replenishment_fdc['item_sku_id'] = replenishment_fdc.index
            return [replenishment_rdc, replenishment_fdc]

        # simple rule: no replenishment and transshipment at all
        inventory_decision = np.zeros((6, 1000)).astype(int)

        return inventory_decision


    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv.append(end_day_inventory)

    def some_other_functions():
        pass
