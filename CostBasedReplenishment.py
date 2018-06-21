"""
Cost Based Replenishment Algorithm

Number of days per Replenishment:
r_star = 7

Function s_star：
Calculating the value of S_star given a value of R_star

Function r_optimizer:
Given a range of values R can take, outputs the best possible R?


"""
from math import sqrt
from scipy.stats import norm

r_star = 7

def z_alpha(h, g, r_star):
    alpha = 1 - (h * r_star)/(h * r_star + g)
    return norm.ppf(alpha)

def s_star(muD, sdD, h, g):
    s = muD * (r_star + 7) + z_alpha(h, g, r_star) * math.sqrt((r_star + muVLT) * (sdD ** 2))
    return s


#def r_star(k, g, h, mud, Es):
#    return math.sqrt((2 * (k + g * Es))/(h * mud))
#
#def s_star(F_1, h, R, g):
#    return F_1(1 - ((h * R)/(h * R + g)))
