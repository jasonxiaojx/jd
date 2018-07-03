#!/usr/bin/env python
# -*- coding:UTF-8 -*-
'''
sample submission for 2nd round
'''

'''
sample submission
参赛者提交代码示例
入参出参已公布
'''
import pandas as pd
import numpy as np
import datetime
from scipy import stats
#from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.ensemble import GradientBoostingRegressor
import multiprocess as mp
import functools

class UserPolicy:
    def __init__(self, initial_inventory, sku_cost):
        sku_demand_fdc_mar = pd.read_csv('mar_pred_fdc.csv')
        sku_demand_rdc_mar = pd.read_csv('mar_pred_rdc.csv')
        sku_demand_fdc_april = pd.read_csv('apr_pred_fdc.csv')
        sku_demand_rdc_april = pd.read_csv('apr_pred_rdc.csv')

        self.sku_demand_rdc_mar = sku_demand_rdc_mar
        sku_demand_rdc_mar['dc_id'] = 0
        sku_demand_mar = sku_demand_rdc_mar.append(sku_demand_fdc_mar)
        self.sku_demand_mar = sku_demand_mar
        
        self.sku_demand_rdc_april = sku_demand_rdc_april
        sku_demand_rdc_april['dc_id'] = 0
        sku_demand_april = sku_demand_rdc_april.append(sku_demand_fdc_april)
        self.sku_demand_april = sku_demand_april
        
        self.inv = initial_inventory
        self.costs = sku_cost
        self.extra_shipping_cost_per_unit = 0.01
        self.fixed_replenish_cost = 0.01
        self.sku_limit = np.asarray([300, 300, 300, 300, 300])
        self.capacity_limit = np.asarray([4000, 5000, 6000, 2000, 1000])
        self.abandon_rate = np.asarray([1./100, 7./100, 10./100, 9./100, 8./100])
        self.intransit = np.zeros((1000,7))


    def daily_decision(self,t):
        '''
        daily decision of inventory allocation
        input values:
            t, decision date
        return values:
            inventory decision, 2-D numpy array, shape (6,1000), type integer
        '''

        # Your algorithms
        std_fc = 1.2
        sku_limit = self.sku_limit
        capacity_limit = self.capacity_limit
        abandon_rate = self.abandon_rate
        inventory = self.inv
        sku_cost = self.costs
        intransit = self.intransit

        if t >= 31:
            sku_demand_dist = self.sku_demand_april
            sku_demand_rdc = self.sku_demand_rdc_april
        else:
            sku_demand_dist = self.sku_demand_mar
            sku_demand_rdc = self.sku_demand_rdc_mar
            
        
        #REPLENISHMENT
        sku_demand_rdc = sku_demand_rdc.sort_values(by='item_sku_id').set_index('item_sku_id')
        sku_cost_r = sku_cost.sort_values(by='item_sku_id').set_index('item_sku_id')
        replen_matrix = sku_cost_r.join(sku_demand_rdc, sort = True)
        rdc_stock = inventory[inventory['dc_id'] == 0][['item_sku_id','stock_quantity']].sort_values(by='item_sku_id').set_index('item_sku_id')

        r_star = 7
        def s_star_m(m):
            def z_alpha(h, g, r_star):
                alpha = 1 - (h * r_star)/(h * r_star + g)
                return stats.norm.ppf(alpha)

            def s_star(muD, sdD, h, g):
                s = muD * (r_star + 7) + z_alpha(h, g, r_star) * ((r_star + 7) * ((sdD * std_fc) ** 2)) ** 0.5
                return s
            output = m.apply(lambda x: s_star(x.sku_avg, x.sku_std, x.holding_cost, x.stockout_cost), axis = 1)
            return output


        def init_replen(s_star, stock, intransit):
            if s_star <= stock + intransit:
                return 0
            else:
                return s_star - stock - intransit


        def replenishment(m,rdc_stock):
            joined = pd.concat([s_star_m(m), rdc_stock, pd.Series(intransit.sum(axis=1),index=[i for i in range(1,1001)])], axis=1)
            joined.columns = ['s_star', 'stock_quantity','intransit']
            output = joined.apply(lambda x: init_replen(x.s_star, x.stock_quantity, x.intransit), axis = 1)
            self.intransit[:,(t-1)%r_star] = np.array(output.get_values())
            replenish_val = np.reshape(output.get_values(),(1000,1)).T
            print(replenish_val.sum(axis=1))
            return replenish_val

        # def replenish():
        #     return np.zeros((1,1000))+100


        #ALLOCATION
        def allocation_rdc(inventory,t):

            dm_invntry = pd.merge(sku_demand_dist, inventory, how='inner', on=('item_sku_id','dc_id'))
            # 怎么利用补货决策修改days_btw? 暂时先假设所有SKU都是每7（R*=7）天到货一次
            #replenish['flag_d'] = replenish.date.apply(lambda x: 1 if x>=t else 0)
            #next_repln = pd.DataFrame(replenish[replenish['flag_d']==1]['date'].groupby(by=replenish.item_sku_id).min()).reset_index()
            #dm_invntry = pd.merge(dm_invntry, next_repln, how='left', on=('item_sku_id'))
            #dm_invntry['days_btw'] = dm_invntry['date'] - t + 1
            #dm_invntry['days_btw'].fillna(31-t, inplace=True)

            dm_invntry['days_btw'] =(7 - t % 7) + 1

            dm_invntry['sd'] = dm_invntry['sku_std'] * std_fc * dm_invntry['days_btw']**0.5
            dm_invntry['exp_sales'] = dm_invntry['sku_avg'] * dm_invntry['days_btw']

            Q = dict(dm_invntry[dm_invntry['dc_id']==0]['stock_quantity'].groupby(by=dm_invntry.item_sku_id,sort=False).sum())
            # 考虑不同的FDC对应的应分配量不一样，应该与其销量损失比例有关
            dm_invntry['Q'] = dm_invntry.item_sku_id.apply(lambda x: Q[x] * 0.68)

            Z = dict((dm_invntry['Q'].groupby(by=dm_invntry.item_sku_id).mean() +
                    dm_invntry['stock_quantity'].groupby(by=dm_invntry.item_sku_id).sum() -
                    dm_invntry['exp_sales'].groupby(by=dm_invntry.item_sku_id).sum()) /
                    dm_invntry['sd'].groupby(by=dm_invntry.item_sku_id).sum())
            dm_invntry['Z'] = dm_invntry.item_sku_id.apply(lambda x: Z[x])
            dm_invntry['capacity'] = (dm_invntry['exp_sales'] +
                                        dm_invntry['Z'] *
                                        dm_invntry['sd'] -
                                        dm_invntry['stock_quantity'])

            while dm_invntry[dm_invntry['capacity']<0]['capacity'].count()>0:
                dm_invntry['flag'] = dm_invntry.capacity.apply(lambda x: 1 if x>0 else 0)
                Z = dict((dm_invntry['Q'].groupby(by=dm_invntry.item_sku_id).mean() +
                    dm_invntry[dm_invntry['flag']==1]['stock_quantity'].groupby(by=dm_invntry.item_sku_id).sum() -
                    dm_invntry[dm_invntry['flag']==1]['exp_sales'].groupby(by=dm_invntry.item_sku_id).sum()) /
                    dm_invntry[dm_invntry['flag']==1]['sd'].groupby(by=dm_invntry.item_sku_id).sum())
                dm_invntry['Z'] = dm_invntry.item_sku_id.apply(lambda x: Z[x])
                dm_invntry['capacity'] = (dm_invntry[dm_invntry['flag']==1]['exp_sales'] +
                                            dm_invntry[dm_invntry['flag']==1]['Z'] *
                                            dm_invntry[dm_invntry['flag']==1]['sd'] -
                                            dm_invntry[dm_invntry['flag']==1]['stock_quantity'])

            dm_invntry['capacity'] = dm_invntry.capacity.apply(lambda x: x if x>0 else 0)

            allocation_rdc = dm_invntry.loc[:,['item_sku_id','dc_id','capacity']].copy()
            return allocation_rdc



        def allocation_fdc_i(init_cr,sku_limit,capacity_limit,inventory,allocation_rdc_r,i):
            import pandas as pd
            import numpy as np
            from scipy import stats

            m_cost_sku_cr = init_cr
            m_cost_sku_cr_cp = init_cr
            step_cr = 0.1
            step_cr_cp = 0.1
            cp_lm = capacity_limit[i-1]
            sku_lm = sku_limit[i-1]
            sku_dm_fdc = sku_demand_dist[sku_demand_dist['dc_id']==i].copy()
            invntry_fdc = inventory[inventory['dc_id']==i].copy()
            dm_invntry_fdc_t = pd.merge(sku_dm_fdc, invntry_fdc, how='inner', on=('item_sku_id','dc_id'))

            dm_invntry_fdc_t = pd.merge(dm_invntry_fdc_t, allocation_rdc_r, how='inner', on=('item_sku_id','dc_id'))
            dm_invntry_fdc_t = pd.merge(dm_invntry_fdc_t, sku_cost, how='inner', on=('item_sku_id'))

            dm_invntry_fdc_t['cost_adj'] = dm_invntry_fdc_t['stockout_cost'] * abandon_rate[i-1] + 0.01 * (1 - abandon_rate[i-1]) + dm_invntry_fdc_t['holding_cost']
            dm_invntry_fdc_t['m_cost'] = dm_invntry_fdc_t.cost_adj.max()

            sku_cnt_fdc_t = 0
            capacity_cnt_fdc_t = 0
            k = 0
            while step_cr > 0.001:
                step_cr = step_cr/2
                while True:
                    k += 1
                    #print('FDC',i,'m_cost_sku_cr',m_cost_sku_cr,'step_cr',step_cr,'sku_cnt_fdc_t',sku_cnt_fdc_t,'sku_lm',sku_lm,'capacity_cnt_fdc_t',capacity_cnt_fdc_t,'cp_lm',cp_lm)
                    m_cost_sku_cr += step_cr
                    dm_invntry_fdc_t['m_cost_sku_cr'] = m_cost_sku_cr
                    dm_invntry_fdc_t['cr'] = 1-dm_invntry_fdc_t['m_cost'] / dm_invntry_fdc_t['cost_adj'] * (1-dm_invntry_fdc_t['m_cost_sku_cr'])
                    dm_invntry_fdc_t['cr'] = dm_invntry_fdc_t['cr'].apply(lambda x: 1 if x>1 else (0 if x<0 else x))

                    dm_invntry_fdc_t['exp_sales'] = stats.norm.ppf(dm_invntry_fdc_t['cr'],loc=dm_invntry_fdc_t['sku_avg']*2,scale=dm_invntry_fdc_t['sku_std'] * std_fc * 2**0.5)

                    dm_invntry_fdc_t['ofstock'] = dm_invntry_fdc_t['exp_sales'] - dm_invntry_fdc_t['stock_quantity']
                    dm_invntry_fdc_t['ofstock'] = dm_invntry_fdc_t.ofstock.apply(lambda x: x if x>0 else 0)
                    dm_invntry_fdc_t['ofstock'] = dm_invntry_fdc_t[['ofstock','capacity']].min(axis=1)
                    if k==1:
                        dm_invntry_fdc_t['rank'] = dm_invntry_fdc_t[dm_invntry_fdc_t['ofstock']>0]['cost_adj'].rank(method='first', ascending=False)
                        dm_invntry_fdc_t['flag'] = dm_invntry_fdc_t['rank'].apply(lambda x: 1 if x<=sku_lm else 0)
                        dm_invntry_fdc_t['ofstock'] = dm_invntry_fdc_t['ofstock']*dm_invntry_fdc_t['flag']

                    sku_cnt_fdc_t = dm_invntry_fdc_t[dm_invntry_fdc_t['ofstock']>0]['ofstock'].count()
                    capacity_cnt_fdc_t = dm_invntry_fdc_t[dm_invntry_fdc_t['ofstock']>0]['ofstock'].sum()

                    if (sku_cnt_fdc_t<=sku_lm) and (capacity_cnt_fdc_t<=cp_lm):
                        dm_invntry_fdc_i = dm_invntry_fdc_t.copy()

                    if (sku_cnt_fdc_t>sku_lm) or (capacity_cnt_fdc_t>cp_lm):
                        sku_flag = (sku_cnt_fdc_t>sku_lm)
                        capacity_flag = (capacity_cnt_fdc_t>cp_lm)
                        dm_invntry_fdc_t = dm_invntry_fdc_i.copy()
                        m_cost_sku_cr -= step_cr
                        break

            if (capacity_flag == True):
                dm_invntry_fdc_i['allocation'] = dm_invntry_fdc_i['ofstock']
                dm_invntry_fdc_cp = dm_invntry_fdc_i.copy()
            else:
                k = 0
                dm_invntry_fdc_i['flag1'] = dm_invntry_fdc_i.ofstock.apply(lambda x: 1 if x>0 else 0)
                dm_invntry_fdc_i['m_cost_cp'] = dm_invntry_fdc_i[dm_invntry_fdc_i['flag1']==1].cost_adj.max()

                while step_cr_cp > 0.001:
                    step_cr_cp = step_cr_cp/2
                    while True:
                        #print('FDC',i,'m_cost_sku_cr',m_cost_sku_cr,'step_cr',step_cr,'sku_cnt_fdc_t',sku_cnt_fdc_t,'sku_lm',sku_lm,'capacity_cnt_fdc_t',capacity_cnt_fdc_t,'cp_lm',cp_lm)
                        k += 1
                        m_cost_sku_cr_cp += step_cr_cp
                        dm_invntry_fdc_i['m_cost_sku_cr_cp'] = m_cost_sku_cr_cp
                        dm_invntry_fdc_i['cr_cp'] = 1-dm_invntry_fdc_i['m_cost_cp']/dm_invntry_fdc_i['cost_adj']*(1-dm_invntry_fdc_i['m_cost_sku_cr_cp'])
                        dm_invntry_fdc_i['cr_cp'] = dm_invntry_fdc_i['cr_cp'].apply(lambda x: 1 if x>1 else (0 if x<0 else x))

                        dm_invntry_fdc_i['exp_sales_cp'] = stats.norm.ppf(dm_invntry_fdc_i['cr_cp'],loc=dm_invntry_fdc_i['sku_avg']*2,scale=dm_invntry_fdc_i['sku_std'] * std_fc * 2**0.5)

                        dm_invntry_fdc_i['ofstock_cp'] = dm_invntry_fdc_i['exp_sales_cp'] - dm_invntry_fdc_i['stock_quantity']
                        dm_invntry_fdc_i['ofstock_cp'] = dm_invntry_fdc_i.ofstock_cp.apply(lambda x: x if x>0 else 0)
                        dm_invntry_fdc_i['ofstock_cp'] = dm_invntry_fdc_i['ofstock_cp'] * dm_invntry_fdc_i['flag1']
                        dm_invntry_fdc_i['ofstock_cp'] = dm_invntry_fdc_i[['ofstock_cp','capacity']].min(axis=1)
                        if k==1:
                            dm_invntry_fdc_i['rank_cp'] = dm_invntry_fdc_i[dm_invntry_fdc_i['ofstock_cp']>0]['cost_adj'].rank(method='first', ascending=False)
                            dm_invntry_fdc_i['flag_cp'] = dm_invntry_fdc_i['rank_cp'].apply(lambda x: 1 if x<=sku_lm else 0)
                            dm_invntry_fdc_i['ofstock_cp'] = dm_invntry_fdc_i['ofstock_cp']*dm_invntry_fdc_i['flag_cp']

                        capacity_cnt_fdc_i = dm_invntry_fdc_i[dm_invntry_fdc_i['ofstock_cp']>0]['ofstock_cp'].sum()

                        if (capacity_cnt_fdc_i<=cp_lm):
                            dm_invntry_fdc_cp = dm_invntry_fdc_i.copy()

                        if (capacity_cnt_fdc_i>cp_lm):
                            dm_invntry_fdc_i = dm_invntry_fdc_cp.copy()
                            m_cost_sku_cr_cp -= step_cr_cp
                            break

                dm_invntry_fdc_cp['allocation'] = dm_invntry_fdc_cp[['ofstock_cp','capacity']].min(axis=1)
                dm_invntry_fdc_cp['allocation'].fillna(0, inplace=True)

            allocation_fdc_cp = dm_invntry_fdc_cp.loc[:,['item_sku_id','dc_id','allocation']].copy()
            return allocation_fdc_cp

        # 提交时多线程要用Multiprocessing的包
        def allocation_fdc(init_cr,sku_limit,capacity_limit,inventory,allocation_rdc_r):
            partial_param = functools.partial(allocation_fdc_i,init_cr,sku_limit,capacity_limit,inventory,allocation_rdc_r)
            pool = mp.Pool(processes=5)
            #pool = Pool(processes=5)
            try:
                fdc_output = pool.map(partial_param, range(1,6))
                pool.close()
                #pool.join()
            except KeyboardInterrupt as e:
                pool.terminate()
                raise e
            temp = fdc_output[0]
            for j in range(1,5):
                temp = temp.append(fdc_output[j])
            return temp

        # 调用函数
        self.replenish = replenishment(replen_matrix,rdc_stock)
        # self.replenish = replenish()
        replenish = self.replenish
        allocation_rdc_r = allocation_rdc(inventory,t)
        allocation_fdc_r = allocation_fdc(0.01,sku_limit,capacity_limit,inventory,allocation_rdc_r)

        distribution = pd.merge(allocation_fdc_r, allocation_rdc_r, how='inner', on=('item_sku_id','dc_id'))
        distribution['daily_decision'] = distribution[['allocation','capacity']].min(axis=1)
        distribution_r = distribution.sort_values(['dc_id', 'item_sku_id'], ascending=[True, True])

        inventory_decision = np.zeros((1000, 5))
        for i in range(0,5):
            inventory_decision[:,i] = distribution_r.loc[distribution_r['dc_id']==i+1]['daily_decision'].values
        inventory_decision = np.transpose(inventory_decision)
        inventory_decision = np.concatenate((replenish, inventory_decision), axis=0)

        # simple rule: no transshipment at all
        # transshipment_decision = np.zeros((5, 1000))
        # transshipment_decision = np.repeat([np.floor(self.inv[0]/self.inv.shape[0])], self.inv.shape[0] -1, axis = 0)
        return inventory_decision.astype(int)


    def info_update(self,end_day_inventory,t):
        '''
        input values: inventory information at the end of day t
        '''
        self.inv = end_day_inventory

    def some_other_functions():
        pass
