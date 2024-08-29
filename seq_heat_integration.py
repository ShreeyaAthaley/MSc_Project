import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress CUDA warnings from tensorflow

#from omlt import OmltBlock, OffsetScaling
#from omlt.io.keras import load_keras_sequential
#from omlt.neuralnet import FullSpaceSmoothNNFormulation
import pyomo.environ as pyo
#import pandas as pd
#import tensorflow.keras as keras

#from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation, \
#FullSpaceSmoothNNFormulation, ReducedSpaceSmoothNNFormulation, ReluBigMFormulation,\
#ReluComplementarityFormulation, ReluPartitionFormulation

#from omlt.neuralnet.activations import ComplementarityReLUActivation
#from omlt.io.keras import keras_reader
#import omlt

#from pyomo.environ import *

#%%
# first, create the Pyomo model
m = pyo.ConcreteModel()
#%%
# sets
m.I = pyo.RangeSet(7) 
m.IP = pyo.Set(within=m.I, initialize=[1,2,3,4,5,6,7])#Process Streams
m.IHU = pyo.Set(within=m.I, initialize=[7]) #Utilities

m.J = pyo.RangeSet(5) 
m.JCU = pyo.Set(within=m.J, initialize=[4,5]) #Utilities

m.K = pyo.Set(initialize=range(11)) # Interval boundaries: no. of streams+utilities
m.KI = pyo.Set(within=m.K, initialize=[1,2,3,4,5,6,7,8,9,10]) #Intervals: KI=K/{0}
#%%
#cost
m.mu_H=pyo.Param(m.IHU,initialize={7:0.25})
m.mu_C=pyo.Param(m.JCU,initialize={4:0.02125, 5:0.2739})

#%%
#bounds
lb1 = {1: 403, 2: 382, 3:384, 4: 558, 5: 327, 6:327, 7: 523.15}
ub1 = {1: 423, 2: 402, 3:404, 4: 578, 5: 347, 6:347, 7: 523.15}
def fb1(m, i):
    return (lb1[i], ub1[i])

#TOUT_H
lb3 = {1: 283, 2: 284, 3:295, 4: 297, 5: 295, 6:326, 7: 522.15}
ub3 = {1: 303, 2: 304, 3:315, 4: 317, 5: 315, 6:346, 7: 522.15}
def fb3(m, i):
    return (lb3[i], ub3[i])
    
#TIN_C
lb2 = {1: 328, 2: 293, 3:367, 4:293.15, 5:248.15}
ub2 = {1: 348, 2: 313, 3:387, 4:293.15, 5:248.15}
def fb2(m, j):
    return (lb2[j], ub2[j])
#TOUT_C
lb4 = {1: 500, 2: 342, 3:368, 4:298.15, 5:249.15}
ub4 = {1: 520, 2: 362, 3:388, 4:298.15, 5:249.15}
def fb4(m, j):
    return (lb4[j], ub4[j])
 
#inlet and outlet temperatures
m.TIN_H = pyo.Var(m.I, bounds=fb1, initialize={1: 413.0987, 2: 392.4241, 3: 394.5512, 4:568.125, 5: 337.4949, 6:337.4949, 7: 523.15})
m.TIN_C = pyo.Var(m.J, bounds=fb2, initialize={1: 338.8852, 2: 303.0903, 3: 377.6938, 4:293.15, 5:248.15})
m.TOUT_H = pyo.Var(m.I, bounds=fb3, initialize={1: 293.1582, 2: 294.3248, 3:305.552, 4: 307.0089, 5: 305.4271, 6:336.4959, 7: 522.15})
m.TOUT_C = pyo.Var(m.J, bounds=fb4, initialize={1: 510.6418, 2: 352.7256, 3: 378.6938, 4:298.15, 5:249.15})

#temperature on interval boundaries
m.T_k= pyo.Var(m.K, bounds=[0,1000],initialize={0:550,1:500,2:450,3:400,4:350,5:300,6:250,7:200,8:190,9:150,10:130})

#%%
m.THIN_link1=pyo.Constraint(expr=m.TIN_H[1]==139.9487+273.15)
m.THIN_link2=pyo.Constraint(expr=m.TIN_H[2]==119.2741+273.15)
m.THIN_link3=pyo.Constraint(expr=m.TIN_H[3]==121.4012+273.15)
m.THIN_link4=pyo.Constraint(expr=m.TIN_H[4]==294.9750+273.15)
m.THIN_link5=pyo.Constraint(expr=m.TIN_H[5]==64.3449+273.15)
m.THIN_link6=pyo.Constraint(expr=m.TIN_H[6]==64.3449+273.15)

m.THOUT_link1=pyo.Constraint(expr=m.TOUT_H[1]==20.0082+273.15)
m.THOUT_link2=pyo.Constraint(expr=m.TOUT_H[2]==21.1748+273.15)
m.THOUT_link3=pyo.Constraint(expr=m.TOUT_H[3]==32.4020+273.15)
m.THOUT_link4=pyo.Constraint(expr=m.TOUT_H[4]==33.8589+273.15)
m.THOUT_link5=pyo.Constraint(expr=m.TOUT_H[5]==32.2771+273.15)
m.THOUT_link6=pyo.Constraint(expr=m.TOUT_H[6]==64.3449+273.151-1)

m.TCIN_link1=pyo.Constraint(expr=m.TIN_C[1]==65.7352+273.15)
m.TCIN_link2=pyo.Constraint(expr=m.TIN_C[2]==29.9403+273.15)
m.TCIN_link3=pyo.Constraint(expr=m.TIN_C[3]==104.5438+273.15)

m.TCOUT_link1=pyo.Constraint(expr=m.TOUT_C[1]==237.4918+273.15)
m.TCOUT_link2=pyo.Constraint(expr=m.TOUT_C[2]==79.5756+273.15)
m.TCOUT_link3=pyo.Constraint(expr=m.TOUT_C[3]==104.5438+273.15+1)
#%%
#disagregated temperatures
m.TIN_HD = pyo.Var(m.I, m.K, bounds=[0,1000], initialize={
    (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0,(1, 5): 0, (1, 6): 0, (1, 7): 0, (1, 8): 0, (1, 9): 0, (1, 10): 0,
    (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 0,(2, 5): 0, (2, 6): 0, (2, 7): 0, (2, 8): 0, (2, 9): 0, (2, 10): 0,
    (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 0,(3, 5): 0, (3, 6): 0, (3, 7): 0, (3, 8): 0, (3, 9): 0, (3, 10): 0,
    (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0,(4, 5): 0, (4, 6): 0, (4, 7): 0, (4, 8): 0, (4, 9): 0, (4, 10): 0,
    (5, 0): 0, (5, 1): 0, (5, 2): 0, (5, 3): 0, (5, 4): 0,(5, 5): 0, (5, 6): 0, (5, 7): 0, (5, 8): 0, (5, 9): 0, (5, 10): 0,
    (6, 0): 0, (6, 1): 0, (6, 2): 0, (6, 3): 0, (6, 4): 0,(6, 5): 0, (6, 6): 0, (6, 7): 0, (6, 8): 0, (6, 9): 0, (6, 10): 0,
    (7, 0): 0, (7, 1): 0, (7, 2): 0, (7, 3): 0, (7, 4): 0,(7, 5): 0, (7, 6): 0, (7, 7): 0, (7, 8): 0, (7, 9): 0, (7, 10): 0})
m.TIN_CD = pyo.Var(m.J, m.K, bounds=[0,1000], initialize={
    (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0,(1, 5): 0, (1, 6): 0, (1, 7): 0, (1, 8): 0, (1, 9): 0, (1, 10): 0,
    (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 0,(2, 5): 0, (2, 6): 0, (2, 7): 0, (2, 8): 0, (2, 9): 0, (2, 10): 0,
    (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 0,(3, 5): 0, (3, 6): 0, (3, 7): 0, (3, 8): 0, (3, 9): 0, (3, 10): 0,
    (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0,(4, 5): 0, (4, 6): 0, (4, 7): 0, (4, 8): 0, (4, 9): 0, (4, 10): 0,
    (5, 0): 0, (5, 1): 0, (5, 2): 0, (5, 3): 0, (5, 4): 0,(5, 5): 0, (5, 6): 0, (5, 7): 0, (5, 8): 0, (5, 9): 0, (5, 10): 0})
m.TOUT_HD = pyo.Var(m.I, m.KI, bounds=[0,1000], initialize={
    (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0,(1, 5): 0, (1, 6): 0, (1, 7): 0, (1, 8): 0, (1, 9): 0, (1, 10): 0, 
    (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 0,(2, 5): 0, (2, 6): 0, (2, 7): 0, (2, 8): 0, (2, 9): 0, (2, 10): 0, 
    (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 0,(3, 5): 0, (3, 6): 0, (3, 7): 0, (3, 8): 0, (3, 9): 0, (3, 10): 0, 
    (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0,(4, 5): 0, (4, 6): 0, (4, 7): 0, (4, 8): 0, (4, 9): 0, (4, 10): 0, 
    (5, 1): 0, (5, 2): 0, (5, 3): 0, (5, 4): 0,(5, 5): 0, (5, 6): 0, (5, 7): 0, (5, 8): 0, (5, 9): 0, (5, 10): 0, 
    (6, 1): 0, (6, 2): 0, (6, 3): 0, (6, 4): 0,(6, 5): 0, (6, 6): 0, (6, 7): 0, (6, 8): 0, (6, 9): 0, (6, 10): 0, 
    (7, 1): 0, (7, 2): 0, (7, 3): 0, (7, 4): 0,(7, 5): 0, (7, 6): 0, (7, 7): 0, (7, 8): 0, (7, 9): 0, (7, 10): 0})
m.TOUT_CD = pyo.Var(m.J, m.KI, bounds=[0,1000], initialize={
    (1, 1): 0, (1, 2): 0, (1, 3): 0, (1, 4): 0,(1, 5): 0, (1, 6): 0, (1, 7): 0, (1, 8): 0, (1, 9): 0, (1, 10): 0, 
    (2, 1): 0, (2, 2): 0, (2, 3): 0, (2, 4): 0,(2, 5): 0, (2, 6): 0, (2, 7): 0, (2, 8): 0, (2, 9): 0, (2, 10): 0, 
    (3, 1): 0, (3, 2): 0, (3, 3): 0, (3, 4): 0,(3, 5): 0, (3, 6): 0, (3, 7): 0, (3, 8): 0, (3, 9): 0, (3, 10): 0, 
    (4, 1): 0, (4, 2): 0, (4, 3): 0, (4, 4): 0,(4, 5): 0, (4, 6): 0, (4, 7): 0, (4, 8): 0, (4, 9): 0, (4, 10): 0, 
    (5, 1): 0, (5, 2): 0, (5, 3): 0, (5, 4): 0,(5, 5): 0, (5, 6): 0, (5, 7): 0, (5, 8): 0, (5, 9): 0, (5, 10): 0})

#minimum approach temperature
m.MAT = pyo.Var(bounds=[10,35],initialize=10)

# binary variables
m.XH_ik=pyo.Var(m.I, m.K, domain=pyo.Binary, initialize=0)
m.YH_ik=pyo.Var(m.I, m.KI, domain=pyo.Binary, initialize=0)
m.ZH_ik=pyo.Var(m.I, m.KI, domain=pyo.Binary, initialize=0)

m.XC_jk=pyo.Var(m.J, m.K, domain=pyo.Binary, initialize=0)
m.YC_jk=pyo.Var(m.J, m.KI, domain=pyo.Binary, initialize=0)
m.ZC_jk=pyo.Var(m.J, m.KI, domain=pyo.Binary, initialize=0)
#%%
# other parameters 
m.alpha_i=pyo.Param(m.I, initialize=600) #=ub1
m.alpha_j=pyo.Param(m.J, initialize=600) #=ub2
m.beta_i=pyo.Param(m.I, initialize=600) #ub3
m.beta_j=pyo.Param(m.J, initialize=600) #ub4

#%%
#CREATING DYNAMIC TEMPERATURE GRID #change the value of k iteratively
def increasing_constraint_rule(m, k):
    if k == 10:
        return pyo.Constraint.Skip
    return m.T_k[k] >= m.T_k[k+1]
m.increasing_constraint = pyo.Constraint(m.K, rule=increasing_constraint_rule)

#%%
#FOR T_IN
# disagregated hot stream temperatures
def TINH_rule1(m, i):
    return m.TIN_H[i] == sum(m.TIN_HD[i, k] for k in m.K)
m.TINH_constraint1 = pyo.Constraint(m.I, rule=TINH_rule1)

def TINH_rule2(m, i, k):
    return m.TIN_HD[i,k] >= m.T_k[k]-m.alpha_i[i]*(1-m.XH_ik[i,k])
m.TINH_constraint2 = pyo.Constraint(m.I, m.K, rule=TINH_rule2)

def TINH_rule3(m, i, k):
    return m.TIN_HD[i,k] <= m.T_k[k]
m.TINH_constraint3 = pyo.Constraint(m.I, m.K, rule=TINH_rule3)

def TINH_rule4(m, i, k):
    return m.TIN_HD[i,k] <= m.alpha_i[i]*m.XH_ik[i,k]
m.TINH_constraint4 = pyo.Constraint(m.I, m.K, rule=TINH_rule4)
#%%
# disagregated cold stream temperatures
def TINC_rule1(m, j):
    return m.TIN_C[j] == sum(m.TIN_CD[j, k] for k in m.K)
m.TINC_constraint1 = pyo.Constraint(m.J, rule=TINC_rule1)

def TINC_rule2(m, j, k):
    return m.TIN_CD[j,k] >= (m.T_k[k]-m.MAT)-(m.alpha_j[j]-m.MAT)*(1-m.XC_jk[j,k])
m.TINC_constraint2 = pyo.Constraint(m.J, m.K, rule=TINC_rule2)

def TINC_rule3(m, j, k):
    return m.TIN_CD[j,k] <= m.T_k[k]-m.MAT
m.TINC_constraint3 = pyo.Constraint(m.J, m.K, rule=TINC_rule3)

def TINC_rule4(m, j, k):
    return m.TIN_CD[j,k] <= m.alpha_j[j]*m.XC_jk[j,k]
m.TINC_constraint4 = pyo.Constraint(m.J, m.K, rule=TINC_rule4)
#%%
#FOR T_OUT
# disagregated hot stream temperatures 
def TOUTH_rule1(m, i):
    return m.TOUT_H[i] == sum(m.TOUT_HD[i, k] for k in m.KI)
m.TOUTH_constraint1 = pyo.Constraint(m.I, rule=TOUTH_rule1)

def TOUTH_rule2(m, i, k):
    return m.TOUT_HD[i,k] >= m.T_k[k]-m.beta_i[i]*(1-m.YH_ik[i,k])
m.TOUTH_constraint2 = pyo.Constraint(m.I, m.KI, rule=TOUTH_rule2)

def TOUTH_rule3(m, i, k):
    return m.TOUT_HD[i,k] <= m.T_k[k-1]
m.TOUTH_constraint3 = pyo.Constraint(m.I, m.KI, rule=TOUTH_rule3)

def TOUTH_rule4(m, i, k):
    return m.TOUT_HD[i,k] <= m.beta_i[i]*m.YH_ik[i,k]
m.TOUTH_constraint4 = pyo.Constraint(m.I, m.KI, rule=TOUTH_rule4)

#%%
# disagregated cold stream temperatures 
def TOUTC_rule1(m, j):
    return m.TOUT_C[j] == sum(m.TOUT_CD[j, k] for k in m.KI)
m.TOUTC_constraint1 = pyo.Constraint(m.J, rule=TOUTC_rule1)

def TOUTC_rule2(m, j, k):
    return m.TOUT_CD[j,k] >= m.T_k[k]-m.MAT-(m.beta_j[j]-m.MAT)*(1-m.YC_jk[j,k])
m.TOUTC_constraint2 = pyo.Constraint(m.J, m.KI, rule=TOUTC_rule2)

def TOUTC_rule3(m, j, k):
    return m.TOUT_CD[j,k] <= m.T_k[k-1]-m.MAT
m.TOUTC_constraint3 = pyo.Constraint(m.J, m.KI, rule=TOUTC_rule3)

def TOUTC_rule4(m, j, k):
    return m.TOUT_CD[j,k] <= m.beta_j[j]*m.YC_jk[j,k]
m.TOUTC_constraint4 = pyo.Constraint(m.J, m.KI, rule=TOUTC_rule4)

#%%
#constraint on X_H,X_C,Y_H,Y_C
def XH_rule1(m, i):
    return sum(m.XH_ik[i,k] for k in m.K) == 1
m.XH_constraint1 = pyo.Constraint(m.I, rule=XH_rule1)

def YH_rule1(m, i):
    return sum(m.YH_ik[i,k] for k in m.KI) == 1
m.YH_constraint1 = pyo.Constraint(m.I, rule=YH_rule1)

def XC_rule1(m, j):
    return sum(m.XC_jk[j,k] for k in m.K) == 1
m.XC_constraint1 = pyo.Constraint(m.J, rule=XC_rule1)

def YC_rule1(m, j):
    return sum(m.YC_jk[j,k] for k in m.KI) == 1
m.YC_constraint1 = pyo.Constraint(m.J, rule=YC_rule1)

#%%
#Part 2
def ZH_rule1(m, i, k):
    if k==1:
        return m.ZH_ik[i,k]==m.XH_ik[i,k-1]-m.YH_ik[i,k]
    return m.ZH_ik[i,k]==m.ZH_ik[i,k-1]+m.XH_ik[i,k-1]-m.YH_ik[i,k]
m.ZH_constraint1 = pyo.Constraint(m.I, m.KI, rule=ZH_rule1)

def ZC_rule1(m, j, k):
    if k==1:
        return pyo.Constraint.Skip
    return m.ZC_jk[j,k-1]==m.ZC_jk[j,k]+m.XC_jk[j,k-1]-m.YC_jk[j,k-1]
m.ZC_constraint1 = pyo.Constraint(m.J, m.KI, rule=ZC_rule1)

#%%
#for hot duties
m.QH_ik= pyo.Var(m.I, m.KI, bounds=[0,100000000],initialize=1000)
m.QH1_ik= pyo.Var(m.IP, m.KI, bounds=[0,100000000],initialize=0)
m.QH2_ik= pyo.Var(m.IP, m.KI, bounds=[0,100000000],initialize=1000)

#for cold duties
m.QC_jk= pyo.Var(m.J, m.KI, bounds=[0,100000000],initialize=1000)
m.QC1_jk= pyo.Var(m.J, m.KI, bounds=[0,100000000],initialize=0)
m.QC2_jk= pyo.Var(m.J, m.KI, bounds=[0,100000000],initialize=1000)

#for heat cascade
m.R_k = pyo.Var(m.K, bounds=[0,100000000],initialize=1000)

#Bounds on heat #big m constraints #should be equal to upper bound of duties
m.gamma_ik=pyo.Param(m.I, m.KI, initialize=100000001)
m.gamma_jk=pyo.Param(m.J, m.KI, initialize=100000001)
m.gammah_ik=pyo.Param(m.I, m.KI, initialize=100000001)
m.gammah_jk=pyo.Param(m.J, m.KI, initialize=100000001)

#%%
#F_H
lb5 = {1: 21, 2: 21, 3:25, 4: 400,5:500, 6:10000, 7:0}
ub5 = {1: 23, 2: 23, 3:26, 4: 600,5:1500, 6:30000, 7:1000000}
def fb5(m, i):
    return (lb5[i], ub5[i])
#F_C
lb6 = {1: 400, 2:50, 3:10000, 4:0, 5:0}
ub6 = {1: 600, 2:1000, 3:50000, 4:1000000, 5:1000000}
def fb6(m, j):
    return (lb6[j], ub6[j])

#hot and cold flow rates
m.F_H = pyo.Var(m.IP, bounds=fb5, initialize={1: 22.2472, 2: 22.6500, 3:25.3917, 4:540.2646 , 5: 818.5446, 6: 20945.768, 7:10000})
m.F_C = pyo.Var(m.J, bounds=fb6, initialize={1: 535.2123, 2:541.5824  ,3:18680.895, 4:5000, 5: 5000})
#%%
m.FH_link1=pyo.Constraint(expr=m.F_H[1]==2000/3600*40.045)
m.FH_link2=pyo.Constraint(expr=m.F_H[2]==2000/3600*40.77)
m.FH_link3=pyo.Constraint(expr=m.F_H[3]==2000/3600*45.705)
m.FH_link4=pyo.Constraint(expr=m.F_H[4]==56994.945/3600*34.125)
m.FH_link5=pyo.Constraint(expr=m.F_H[5]==1930.555/3600*1526.38)
m.FH_link6=pyo.Constraint(expr=m.F_H[6]==20945.768)#directly equate to predicted duty

m.FC_link1=pyo.Constraint(expr=m.F_C[1]==60504.452/3600*31.845)
m.FC_link2=pyo.Constraint(expr=m.F_C[2]==3832.139/3600*508.775) 
m.FC_link3=pyo.Constraint(expr=m.F_C[3]==18680.895)#directly equate to predicted duty 
#%%
# Duty of H
def QH_rule1(m, i, k):
    return m.QH_ik[i,k] == m.QH1_ik[i,k]+m.QH2_ik[i,k]
m.QH_constraint1 = pyo.Constraint(m.IP, m.KI, rule=QH_rule1)

def QH_rule2(m, i, k):
    return m.QH1_ik[i,k] >= m.F_H[i]*(m.T_k[k-1]-m.T_k[k])-m.gamma_ik[i,k]*(1-m.ZH_ik[i,k])
m.QH_constraint2 = pyo.Constraint(m.IP, m.KI, rule=QH_rule2)

def QH_rule3(m, i, k):
    return m.QH1_ik[i,k] <= m.F_H[i]*(m.T_k[k-1]-m.T_k[k])
m.QH_constraint3 = pyo.Constraint(m.IP, m.KI, rule=QH_rule3)

def QH_rule4(m, i, k):
    return m.QH1_ik[i,k] <= m.gamma_ik[i,k]*(m.ZH_ik[i,k])
m.QH_constraint4 = pyo.Constraint(m.IP, m.KI, rule=QH_rule4)

def QH_rule5(m, i, k):
    return m.QH2_ik[i,k] >= m.F_H[i]*(m.T_k[k-1]-m.TOUT_HD[i,k])-m.gamma_ik[i,k]*(1-m.YH_ik[i,k])
m.QH_constraint5 = pyo.Constraint(m.IP, m.KI, rule=QH_rule5)

def QH_rule6(m, i, k):
    return m.QH2_ik[i,k] <= m.F_H[i]*(m.T_k[k-1]-m.TOUT_HD[i,k])+m.gamma_ik[i,k]*(1-m.YH_ik[i,k])
m.QH_constraint6 = pyo.Constraint(m.IP, m.KI, rule=QH_rule6)

def QH_rule7(m, i, k):
    return m.QH2_ik[i,k] <= m.gamma_ik[i,k]*(m.YH_ik[i,k])
m.QH_constraint7 = pyo.Constraint(m.IP, m.KI, rule=QH_rule7)

#%%
# Duty of C
def QC_rule1(m, j, k):
    return m.QC_jk[j,k] == m.QC1_jk[j,k]+m.QC2_jk[j,k]
m.QC_constraint1 = pyo.Constraint(m.J, m.KI, rule=QC_rule1)

def QC_rule2(m, j, k):
    return m.QC1_jk[j,k] >= m.F_C[j]*(m.T_k[k-1]-m.T_k[k])-m.gamma_jk[j,k]*(1-m.ZC_jk[j,k])
m.QC_constraint2 = pyo.Constraint(m.J, m.KI, rule=QC_rule2)

def QC_rule3(m, j, k):
    return m.QC1_jk[j,k] <= m.F_C[j]*(m.T_k[k-1]-m.T_k[k])
m.QC_constraint3 = pyo.Constraint(m.J, m.KI, rule=QC_rule3)

def QC_rule4(m, j, k):
    return m.QC1_jk[j,k] <= m.gamma_jk[j,k]*(m.ZC_jk[j,k])
m.QC_constraint4 = pyo.Constraint(m.J, m.KI, rule=QC_rule4)

def QC_rule5(m, j, k):
    return m.QC2_jk[j,k] >= m.F_C[j]*(m.TOUT_CD[j,k]-m.T_k[k]+m.MAT)-m.gamma_jk[j,k]*(1-m.YC_jk[j,k])
m.QC_constraint5 = pyo.Constraint(m.J, m.KI, rule=QC_rule5)

def QC_rule6(m, j, k):
    return m.QC2_jk[j,k] <= m.F_C[j]*(m.TOUT_CD[j,k]-m.T_k[k]+m.MAT)+m.gamma_jk[j,k]*(1-m.YC_jk[j,k])
m.QC_constraint6 = pyo.Constraint(m.J, m.KI, rule=QC_rule6)

def QC_rule7(m, j, k):
    return m.QC2_jk[j,k] <= m.gamma_jk[j,k]*(m.YC_jk[j,k])
m.QC_constraint7 = pyo.Constraint(m.J, m.KI, rule=QC_rule7)

#%%
def R_rule2(m):
    k_index = list(m.K)[-1]
    return m.R_k[k_index] == 0 
m.R_constraint2 = pyo.Constraint(rule=R_rule2)

def R_rule3(m):
    k_index = list(m.K)[0]
    return m.R_k[k_index] == 0 
m.R_constraint3 = pyo.Constraint(rule=R_rule3)

#%%
def pinch_rule(m,k):
    return m.R_k[k]-m.R_k[k-1]==sum(m.QH_ik[i,k] for i in m.I)-sum(m.QC_jk[j,k] for j in m.J)
m.pinch_constraint = pyo.Constraint(m.KI,rule=pinch_rule)
#%%
# tightening constraints
def Tight_rule1(m, i, k):
    return m.XH_ik[i,k-1] <= m.YH_ik[i,k] + m.ZH_ik[i,k]
m.Tight_constraint1 = pyo.Constraint(m.I, m.KI, rule=Tight_rule1)

def Tight_rule2(m, j, k):
    return m.XC_jk[j,k] <= m.YC_jk[j,k] + m.ZC_jk[j,k]
m.Tight_constraint2 = pyo.Constraint(m.J, m.KI, rule=Tight_rule2)
#%%
def obj_pinch(m):
    return sum(m.mu_H[i]*sum(m.QH_ik[i,k] for k in m.KI) for i in m.IHU)+sum(m.mu_C[j]*sum(m.QC_jk[j,k] for k in m.KI) for j in m.JCU)
m.obj_func = pyo.Objective(rule=obj_pinch,sense=pyo.minimize)

#%%
#m.obj_func = pyo.Objective(expr=0,sense=minimize)
#%%
solver = pyo.SolverFactory('scip')
solver.options ={'limits/gap': 0.01, 'numerics/feastol': 1e-4, 'numerics/dualfeastol': 1e-4, 'propagating/obbt/dualfeastol': 1e-04}
status = solver.solve(m, tee=True)
#%%
#solver = pyo.SolverFactory('mindtpy')
#solver = pyo.SolverFactory('IPOPT')
#solver.options['tol'] = 1E-5
#solver.options['max_iter'] = 10000
#status = solver.solve(m, tee=True , iteration_limit=10000, mip_solver='glpk',nlp_solver='ipopt', strategy='OA',obj_bound=100000000)
       