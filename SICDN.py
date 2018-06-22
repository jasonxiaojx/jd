"""
Testing data would be evaluated based on a few variable factors:
dijt: Demand for item i at store j on day t.
dRit: Demand for item i at RDC on day t.
Iijt: Inventory for item i at store j on day t.
IRit: Inventory for item i at RDC on day t.
wijt: Number of item i in transit from RDC to store j on day t.
rit: Amount of item i ordered from supplier on day t.
alphaj: Percentage of unmet demand in total demand.
Nj: Maximum number of unique items allowed to be transfered from RDC to FDC j.
Mj: Maxinum number of item count allowed for transfer from RDC to FDC j.

Cost is calculated using:

Ctotal = sum(c1 + c2 + c3 + c4 + c5) across all ij(t if applies)

"""
