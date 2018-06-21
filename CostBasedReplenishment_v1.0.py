"""
Cost Based Replenishment Strategy v1.0:

Might require systematic adjustments

Helper Functions:
s_star_m:
calculates a list of S_STAR valuesself.
z_alpha:
calculates a z-alpha of standard normal distr.
init_replen:
checks if a replenishment is needed, if so how much, if not 0.

Output Function:
replenishment:
Takes a replenishment matrix containing 4-columns in the order
stockout_cost --->	holding_cost --->	sku_avg	--->  sku_std
Also takes a matrix showing the stock of each sku at the RDC
Returns a series with item IDs and the supposed replenishment

"""

import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import norm

r_star = 7
par_RDC_total_sale = pd.read_csv("par_RDC_total_sale.csv").set_index('item_sku_id')
sku_cost = pd.read_csv('sku_cost.csv').set_index('item_sku_id')
sku_stock = pd.read_csv('sku_stock.csv').set_index('item_sku_id')
replen_matrix = sku_cost.join(par_RDC_total_sale, sort = True)
rdc_stock = sku_stock[sku_stock['dc_id'] == 0]['stock_quantity']

def s_star_m(m):
    def z_alpha(h, g, r_star):
        alpha = 1 - (h * r_star)/(h * r_star + g)
        return norm.ppf(alpha)

    def s_star(muD, sdD, h, g):
        s = muD * (r_star + 7) + z_alpha(h, g, r_star) * math.sqrt((r_star + 7) * (sdD ** 2))
        return s
    output = m.apply(lambda x: s_star(x.sku_avg, x.sku_std, x.holding_cost, x.stockout_cost), axis = 1)
    return output

def init_replen(s_star, stock):
    if s_star <= stock:
        return 0
    else:
        return s_star - stock

def replenishment(m,rdc_stock):
    joined = pd.concat([s_star_m(m), rdc_stock], axis=1)
    joined.columns = ['s_star', 'stock_quantity']
    output = joined.apply(lambda x: init_replen(x.s_star, x.stock_quantity), axis = 1)

    return output
