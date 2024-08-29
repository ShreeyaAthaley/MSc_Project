import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress CUDA warnings from tensorflow

from omlt import OmltBlock, OffsetScaling
from omlt.io.keras import load_keras_sequential
import pyomo.environ as pyo
import pandas as pd
import tensorflow.keras as keras

from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation, \
FullSpaceSmoothNNFormulation, ReducedSpaceSmoothNNFormulation, ReluBigMFormulation,\
ReluComplementarityFormulation, ReluPartitionFormulation

from omlt.neuralnet.activations import ComplementarityReLUActivation
from omlt.io.keras import keras_reader
import omlt

from pyomo.environ import *

import numpy as np

#%%
m = pyo.ConcreteModel()
#%%
# create the OmltBlock to hold the neural network model
m.compressor_T1 = OmltBlock()
m.compressor_E1 = OmltBlock()
m.compressor_T2 = OmltBlock()
m.compressor_E2 = OmltBlock()
m.compressor_T3 = OmltBlock()
m.compressor_E3 = OmltBlock()
m.compressor_T4 = OmltBlock()
m.compressor_E4 = OmltBlock()
m.compressor_H2_T = OmltBlock()
m.compressor_H2_E = OmltBlock()

m.mixer_feed = OmltBlock()
m.mixer_reactor = OmltBlock()

m.reactor_comp=OmltBlock()
m.reactor_ch3oh=OmltBlock()
m.reactor_h2=OmltBlock()
m.reactor_T=OmltBlock()
m.reactor_P=OmltBlock()

m.cooler_vap_co2_co=OmltBlock()
m.cooler_vap_h2=OmltBlock()
m.cooler_liq_ch3oh_h2o=OmltBlock()
m.cooler_liq_co=OmltBlock()
m.cooler_power=OmltBlock()
m.cooler_VD=OmltBlock()
m.cooler_LD=OmltBlock()

m.compressor_Mix_T = OmltBlock()
m.compressor_Mix_E = OmltBlock()

m.valve_vap_co = OmltBlock()
m.valve_liq_ch3oh_h2o = OmltBlock()
m.valve_liq_co2 = OmltBlock()
m.valve_T = OmltBlock()
m.valve_VD = OmltBlock()
m.valve_LD = OmltBlock()

m.heater = OmltBlock()

m.dis_vap_co2= OmltBlock()
m.dis_vap_ch3oh = OmltBlock()
m.dis_vap_co=OmltBlock()
m.dis_vap_h2o=OmltBlock()
m.dis_con_T=OmltBlock()
m.dis_reb_T=OmltBlock()
m.dis_con_duty=OmltBlock()
m.dis_reb_duty=OmltBlock()
m.dis_VD=OmltBlock()
m.dis_LD=OmltBlock()
m.dis_max_vap=OmltBlock()
m.dis_no_stage=OmltBlock()
m.dis_feed_stage=OmltBlock()

m.cooler_liq_co2_2=OmltBlock()
m.cooler_liq_ch3oh_2=OmltBlock()
m.cooler_liq_co_2=OmltBlock()
m.cooler_liq_h2o_2=OmltBlock()
m.cooler_power_2=OmltBlock()
m.cooler_VD_2=OmltBlock()
m.cooler_LD_2=OmltBlock()

#%%
#defining inputs and outputs
inputs = ['F1', 'T1', 'P1', 'PR']
inputs2 = ['F2', 'T2', 'P2', 'PR2']
inputs3 = ['F3', 'T3', 'P3', 'PR3']
outputT = ['T2']
outputE = ['E1']
inputs4 = ['F1CO2', 'F1T', 'F2H2', 'F2T', 'P']
inputs5 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'S1T','S2CO2','S2H2','S2CH3OH','S2CO','S2H2O', 'S2T', 'P']
inputs6 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'S1T','S1P', 'V']
inputs7 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'S1T','S1P', 'S1dT']
inputs8 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'T','P', 'PR']
inputs9 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'S1T','S1P', 'S1dP']
inputs10 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'S1T','S1P', 'S1T2']
inputs11 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'S1T','S1P']
inputs12 = ['S1CO2','S1H2','S1CH3OH','S1CO','S1H2O', 'S1T','S1P', 'S1T2']

output_comp = ['CO2','CO','H2O']
output_ch3oh = ['CH3OH']
output_h2=['H2']
output_T=['T']
output_P=['P']

cooler1=['CO2','CO']
cooler3=['H2']
cooler5=['CH3OH','H2O']
cooler9=['E']
cooler10=['VD']
cooler11=['LD']

valve3=['CO']
valve5=['CH3OH','H2O']
valve6=['CO2']
valve8=['T']
valve9=['VD']
valve10=['LD']

dis1=['CO2']
dis2=['CH3OH']
dis3=['CO']
dis4=['H2O']
dis5=['conT']
dis6=['rebT']
dis7=['cE']
dis8=['rE']
dis9=['VD']
dis10=['LD']
dis11=['max_vap']
dis12=['nostages']
dis13=['fstages']

cooler1_2=['CO2']
cooler2_2=['CH3OH']
cooler3_2=['CO']
cooler4_2=['H2O']
cooler5_2=['E']
cooler6_2=['VD']
cooler7_2=['LD']

#%%
# load the Keras model
nn_compressor_T = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_T.keras', compile=False)
nn_compressor_E = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_P.keras', compile=False)
nn_compressor_T_HP = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_T_CO2.keras', compile=False)
nn_compressor_E_HP = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_E_CO2.keras', compile=False)
nn_compressor_T_HP2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_T_H2.keras', compile=False)
nn_compressor_E_HP2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_E_H2.keras', compile=False)

nn_mixer_feed = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\mixer_nn.keras', compile=False)
nn_mixer_reactor = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\mixer_nn_2.keras', compile=False)

nn_reactor_comp = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_major_comp.keras', compile=False)
nn_reactor_ch3oh = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_ch3oh.keras', compile=False)
nn_reactor_h2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_h2.keras', compile=False)
nn_reactor_T = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_T.keras', compile=False)
nn_reactor_P = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_P.keras', compile=False) 

nn_cooler_vap_co2_co = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_vap_co2_co.keras', compile=False) 
nn_cooler_vap_ch3oh = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_vap_ch3oh.keras', compile=False) 
nn_cooler_vap_h2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_vap_h2.keras', compile=False) 
nn_cooler_vap_h2o = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_vap_h2o.keras', compile=False) 
nn_cooler_liq_ch3oh_h2o = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_liq_met_h2o.keras', compile=False) 
nn_cooler_liq_co2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_liq_co2.keras', compile=False) 
nn_cooler_liq_h2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_liq_h2.keras', compile=False)
nn_cooler_liq_co = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_liq_co.keras', compile=False)
nn_cooler_E = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_E.keras', compile=False)
nn_cooler_VD = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_VD.keras', compile=False)
nn_cooler_LD = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_LD.keras', compile=False)

nn_compressor_T_mix= keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_Mix_T.keras', compile=False)
nn_compressor_E_mix= keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\compressor_nn_Mix_E.keras', compile=False)

nn_valve_vap_co2_h2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_vap_co2_h2.keras', compile=False)
nn_valve_vap_ch3oh=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_vap_ch3oh.keras', compile=False)
nn_valve_vap_co=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_vap_co.keras', compile=False)
nn_valve_vap_h2o=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_vap_h2o.keras', compile=False)
nn_valve_liq_ch3oh_h2o=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_liq_ch3oh_h2o.keras', compile=False)
nn_valve_liq_co2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_liq_co2.keras', compile=False)
nn_valve_liq_co=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_liq_co.keras', compile=False)
nn_valve_T=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_T.keras', compile=False)
nn_valve_VD=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_VD.keras', compile=False)
nn_valve_LD=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_LD.keras', compile=False)

nn_heater=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pre_heater_new.keras', compile=False)

nn_dis_vap_co2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_vap_co2.keras', compile=False)
nn_dis_vap_ch3oh=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_vap_ch3oh_2.keras', compile=False)
nn_dis_vap_co=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_vap_co.keras', compile=False)
nn_dis_vap_h2o=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_vap_h2o.keras', compile=False)
nn_dis_con_T=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_con_T.keras', compile=False)
nn_dis_reb_T=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_reb_T.keras', compile=False)
nn_dis_con_duty=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_con_duty.keras', compile=False)
nn_dis_reb_duty=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_reb_duty.keras', compile=False)
nn_dis_VD=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_VD.keras', compile=False)
nn_dis_LD=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_LD.keras', compile=False)
nn_dis_max_vap=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_max_vap_rate.keras', compile=False)
nn_dis_no_stage=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_no_of_stage.keras', compile=False)
nn_dis_feed_stage=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_feed_stage.keras', compile=False)

nn_cooler_liq_co2_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_co2_2.keras', compile=False)
nn_cooler_liq_ch3oh_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_ch3oh_2.keras', compile=False)
nn_cooler_liq_co_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_co_2.keras', compile=False)
nn_cooler_liq_h2o_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_h2o_2.keras', compile=False)
nn_cooler_power_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_energy_2.keras', compile=False)
nn_cooler_VD_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_VD_2.keras', compile=False)
nn_cooler_LD_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_LD_2.keras', compile=False)

#%%
#offset scaling=(Actual input-Lower Bound)/(Upper Bound -Lower Bound) where Lower Bound=offset and Difference =scaling
x_offset={'F1': 1000, 'T1': 20, 'P1':80, 'PR':1.05}
x_factor={'F1': 2000, 'T1': 30, 'P1':270, 'PR':2.95}
x_offset2={'F2': 1000, 'T2': 5, 'P2':800, 'PR2':1.05}
x_factor2={'F2': 2000, 'T2': 45, 'P2':3200, 'PR2':2.95}
x_offset3={'F3': 4000, 'T3': 5, 'P3':1000, 'PR3':1.05}
x_factor3={'F3': 4000, 'T3': 40, 'P3':4000, 'PR3':2.95}
x_offset4={'F1CO2':500, 'F1T':30, 'F2H2':3000, 'F2T':30, 'P':5000}
x_factor4={'F1CO2':3500, 'F1T':270, 'F2H2':6000, 'F2T':270, 'P':5000}
x_offset5={'S1CO2':1000,'S1H2':3000,'S1CH3OH':0,'S1CO':0,'S1H2O':0, 'S1T':30,'S2CO2':1000,'S2H2':3000,'S2CH3OH':0,'S2CO':0,'S2H2O':0, 'S2T':30, 'P':5000}
x_factor5={'S1CO2':9000,'S1H2':47000,'S1CH3OH':350,'S1CO':4000,'S1H2O':100, 'S1T':220,'S2CO2':9000,'S2H2':47000,'S2CH3OH':350,'S2CO':4000,'S2H2O':100, 'S2T':220, 'P':5000}
x_offset6={'S1CO2':2000,'S1H2':35000,'S1CH3OH':20,'S1CO':500,'S1H2O':5, 'S1T':200,'S1P':5000, 'V':30}
x_factor6={'S1CO2':8000,'S1H2':25000,'S1CH3OH':330,'S1CO':3500,'S1H2O':95, 'S1T':100,'S1P':5000, 'V':30}
x_offset7={'S1CO2':800,'S1H2':29000,'S1CH3OH':170,'S1CO':540,'S1H2O':480, 'S1T':200,'S1P':300, 'S1dT':150.8935546875}
x_factor7={'S1CO2':8200,'S1H2':29100,'S1CH3OH':3040,'S1CO':4660,'S1H2O':3270, 'S1T':150,'S1P':9700, 'S1dT':178.623046875}
x_offset8={'S1CO2':2000,'S1H2':20000,'S1CH3OH':20,'S1CO':500,'S1H2O':5, 'T':27,'P':3000, 'PR':1.05}
x_factor8={'S1CO2':8000,'S1H2':40000,'S1CH3OH':330,'S1CO':3500,'S1H2O':95, 'T':28,'P':4000, 'PR':2.95}
x_offset9={'S1CO2':40,'S1H2':10,'S1CH3OH':1000,'S1CO':1,'S1H2O':1000, 'S1T':10.5,'S1P':4500, 'S1dP':4308.074951171875}
x_factor9={'S1CO2':90,'S1H2':80,'S1CH3OH':1600,'S1CO':6,'S1H2O':1600, 'S1T':89.5,'S1P':2000, 'S1dP':2082.391357421875}
x_offset10={'S1CO2':4,'S1H2':0,'S1CH3OH':910,'S1CO':0.005,'S1H2O':960, 'S1T':10,'S1P':100, 'S1T2':40}
x_factor10={'S1CO2':126,'S1H2':1,'S1CH3OH':1690,'S1CO':4.995,'S1H2O':1640, 'S1T':75,'S1P':100, 'S1T2':60}
x_offset11={'S1CO2':4,'S1H2':0,'S1CH3OH':910,'S1CO':0.005,'S1H2O':960, 'S1T':40,'S1P':100}
x_factor11={'S1CO2':126,'S1H2':1,'S1CH3OH':1690,'S1CO':4.995,'S1H2O':1640, 'S1T':60,'S1P':100}
x_offset12={'S1CO2':4,'S1H2':0,'S1CH3OH':900,'S1CO':0.004,'S1H2O':3, 'S1T':60,'S1P':90, 'S1T2':30}
x_factor12={'S1CO2':126,'S1H2':1,'S1CH3OH':1700,'S1CO':5.996,'S1H2O':152, 'S1T':25,'S1P':110, 'S1T2':30}


y_offsetT = {'T2': 0}
y_factorT = {'T2': 1}
y_offsetE = {'E1': 0}
y_factorE = {'E1': 1}

y_offset_comp = {'CO2':0,'CO':0,'H2O':0}
y_offset_ch3oh = {'CH3OH':0}
y_offset_h2={'H2':0}
y_offset_T={'T':0}
y_offset_P={'P':0}
y_factor_comp = {'CO2':1,'CO':1,'H2O':1}
y_factor_ch3oh = {'CH3OH':1}
y_factor_h2={'H2':1}
y_factor_T={'T':1}
y_factor_P={'P':1}

y_offset_cooler1={'CO2':0,'CO':0}
y_offset_cooler3={'H2':0}
y_offset_cooler5={'CH3OH':0,'H2O':0}
y_offset_cooler9={'E':0}
y_offset_cooler10={'VD':0}
y_offset_cooler11={'LD':0}
y_factor_cooler1={'CO2':1,'CO':1}
y_factor_cooler3={'H2':1}
y_factor_cooler5={'CH3OH':1,'H2O':1}
y_factor_cooler9={'E':1}
y_factor_cooler10={'VD':1}
y_factor_cooler11={'LD':1}


y_offset_valve3={'CO':0.00010143259257129787}
y_offset_valve5={'CH3OH':0,'H2O':0}
y_offset_valve6={'CO2':0}
y_offset_valve8={'T':0}
y_offset_valve9={'VD':0}
y_offset_valve10={'LD':0}
y_factor_valve3={'CO':0.0018069152945605898}
y_factor_valve5={'CH3OH':1,'H2O':1}
y_factor_valve6={'CO2':1}
y_factor_valve8={'T':1}
y_factor_valve9={'VD':1}
y_factor_valve10={'LD':1}

y_offset_dis1={'CO2':0}
y_offset_dis2={'CH3OH':0}
y_offset_dis3={'CO':1.4735751681857644e-06}
y_offset_dis4={'H2O':0}
y_offset_dis5={'conT':0}
y_offset_dis6={'rebT':0}
y_offset_dis7={'cE':0}
y_offset_dis8={'rE':0}
y_offset_dis9={'VD':0}
y_offset_dis10={'LD':0}
y_offset_dis11={'max_vap':0}
y_offset_dis12={'nostages':0}
y_offset_dis13={'fstages':0}
y_factor_dis1={'CO2':1}
y_factor_dis2={'CH3OH':1}
y_factor_dis3={'CO':0.0013873306274414065}
y_factor_dis4={'H2O':1}
y_factor_dis5={'conT':1}
y_factor_dis6={'rebT':1}
y_factor_dis7={'cE':1}
y_factor_dis8={'rE':1}
y_factor_dis9={'VD':1}
y_factor_dis10={'LD':1}
y_factor_dis11={'max_vap':1}
y_factor_dis12={'nostages':1}
y_factor_dis13={'fstages':1}

y_offset_cooler1_2={'CO2':0}
y_offset_cooler2_2={'CH3OH':0}
y_offset_cooler3_2={'CO':0}
y_offset_cooler4_2={'H2O':0}
y_offset_cooler5_2={'E':0}
y_offset_cooler6_2={'VD':0}
y_offset_cooler7_2={'LD':0}
y_factor_cooler1_2={'CO2':1}
y_factor_cooler2_2={'CH3OH':1}
y_factor_cooler3_2={'CO':1}
y_factor_cooler4_2={'H2O':1}
y_factor_cooler5_2={'E':1}
y_factor_cooler6_2={'VD':1}
y_factor_cooler7_2={'LD':1}

#%%
# Note: The neural network is in the scaled space. We want access to the variables in the unscaled space. Therefore, we need to tell OMLT about the scaling factors
scalerT = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offsetT[outputT[i]] for i in range(len(outputT))},
        factor_outputs={i: y_factorT[outputT[i]] for i in range(len(outputT))}
    )

scalerE = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offsetE[outputE[i]] for i in range(len(outputE))},
        factor_outputs={i: y_factorE[outputE[i]] for i in range(len(outputE))}
    )

scalerT2 = OffsetScaling(
        offset_inputs={i: x_offset2[inputs2[i]] for i in range(len(inputs2))},
        factor_inputs={i: x_factor2[inputs2[i]] for i in range(len(inputs2))},
        offset_outputs={i: y_offsetT[outputT[i]] for i in range(len(outputT))},
        factor_outputs={i: y_factorT[outputT[i]] for i in range(len(outputT))}
    )

scalerE2 = OffsetScaling(
        offset_inputs={i: x_offset2[inputs2[i]] for i in range(len(inputs2))},
        factor_inputs={i: x_factor2[inputs2[i]] for i in range(len(inputs2))},
        offset_outputs={i: y_offsetE[outputE[i]] for i in range(len(outputE))},
        factor_outputs={i: y_factorE[outputE[i]] for i in range(len(outputE))}
    )

scalerT3 = OffsetScaling(
        offset_inputs={i: x_offset3[inputs3[i]] for i in range(len(inputs3))},
        factor_inputs={i: x_factor3[inputs3[i]] for i in range(len(inputs3))},
        offset_outputs={i: y_offsetT[outputT[i]] for i in range(len(outputT))},
        factor_outputs={i: y_factorT[outputT[i]] for i in range(len(outputT))}
    )

scalerE3 = OffsetScaling(
        offset_inputs={i: x_offset3[inputs3[i]] for i in range(len(inputs3))},
        factor_inputs={i: x_factor3[inputs3[i]] for i in range(len(inputs3))},
        offset_outputs={i: y_offsetE[outputE[i]] for i in range(len(outputE))},
        factor_outputs={i: y_factorE[outputE[i]] for i in range(len(outputE))}
    )

scalerT4 = OffsetScaling(
        offset_inputs={i: x_offset4[inputs4[i]] for i in range(len(inputs4))},
        factor_inputs={i: x_factor4[inputs4[i]] for i in range(len(inputs4))},
        offset_outputs={i: y_offsetT[outputT[i]] for i in range(len(outputT))},
        factor_outputs={i: y_factorT[outputT[i]] for i in range(len(outputT))}
    )

scalerT5 = OffsetScaling(
        offset_inputs={i: x_offset5[inputs5[i]] for i in range(len(inputs5))},
        factor_inputs={i: x_factor5[inputs5[i]] for i in range(len(inputs5))},
        offset_outputs={i: y_offsetT[outputT[i]] for i in range(len(outputT))},
        factor_outputs={i: y_factorT[outputT[i]] for i in range(len(outputT))}
    )

scalerR1 = OffsetScaling(
        offset_inputs={i: x_offset6[inputs6[i]] for i in range(len(inputs6))},
        factor_inputs={i: x_factor6[inputs6[i]] for i in range(len(inputs6))},
        offset_outputs={i: y_offset_comp[output_comp[i]] for i in range(len(output_comp))},
        factor_outputs={i: y_factor_comp[output_comp[i]] for i in range(len(output_comp))}
    )

scalerR2 = OffsetScaling(
        offset_inputs={i: x_offset6[inputs6[i]] for i in range(len(inputs6))},
        factor_inputs={i: x_factor6[inputs6[i]] for i in range(len(inputs6))},
        offset_outputs={i: y_offset_ch3oh[output_ch3oh[i]] for i in range(len(output_ch3oh))},
        factor_outputs={i: y_factor_ch3oh[output_ch3oh[i]] for i in range(len(output_ch3oh))}
    )

scalerR3 = OffsetScaling(
        offset_inputs={i: x_offset6[inputs6[i]] for i in range(len(inputs6))},
        factor_inputs={i: x_factor6[inputs6[i]] for i in range(len(inputs6))},
        offset_outputs={i: y_offset_h2[output_h2[i]] for i in range(len(output_h2))},
        factor_outputs={i: y_factor_h2[output_h2[i]] for i in range(len(output_h2))}
    )

scalerR4 = OffsetScaling(
        offset_inputs={i: x_offset6[inputs6[i]] for i in range(len(inputs6))},
        factor_inputs={i: x_factor6[inputs6[i]] for i in range(len(inputs6))},
        offset_outputs={i: y_offset_T[output_T[i]] for i in range(len(output_T))},
        factor_outputs={i: y_factor_T[output_T[i]] for i in range(len(output_T))}
    )

scalerR5 = OffsetScaling(
        offset_inputs={i: x_offset6[inputs6[i]] for i in range(len(inputs6))},
        factor_inputs={i: x_factor6[inputs6[i]] for i in range(len(inputs6))},
        offset_outputs={i: y_offset_P[output_P[i]] for i in range(len(output_P))},
        factor_outputs={i: y_factor_P[output_P[i]] for i in range(len(output_P))}
    )

scalerC1 = OffsetScaling(
        offset_inputs={i: x_offset7[inputs7[i]] for i in range(len(inputs7))},
        factor_inputs={i: x_factor7[inputs7[i]] for i in range(len(inputs7))},
        offset_outputs={i: y_offset_cooler1[cooler1[i]] for i in range(len(cooler1))},
        factor_outputs={i: y_factor_cooler1[cooler1[i]] for i in range(len(cooler1))}
    )

scalerC3 = OffsetScaling(
        offset_inputs={i: x_offset7[inputs7[i]] for i in range(len(inputs7))},
        factor_inputs={i: x_factor7[inputs7[i]] for i in range(len(inputs7))},
        offset_outputs={i: y_offset_cooler3[cooler3[i]] for i in range(len(cooler3))},
        factor_outputs={i: y_factor_cooler3[cooler3[i]] for i in range(len(cooler3))}
    )

scalerC5 = OffsetScaling(
        offset_inputs={i: x_offset7[inputs7[i]] for i in range(len(inputs7))},
        factor_inputs={i: x_factor7[inputs7[i]] for i in range(len(inputs7))},
        offset_outputs={i: y_offset_cooler5[cooler5[i]] for i in range(len(cooler5))},
        factor_outputs={i: y_factor_cooler5[cooler5[i]] for i in range(len(cooler5))}
    )

scalerC9 = OffsetScaling(
        offset_inputs={i: x_offset7[inputs7[i]] for i in range(len(inputs7))},
        factor_inputs={i: x_factor7[inputs7[i]] for i in range(len(inputs7))},
        offset_outputs={i: y_offset_cooler9[cooler9[i]] for i in range(len(cooler9))},
        factor_outputs={i: y_factor_cooler9[cooler9[i]] for i in range(len(cooler9))}
    )

scalerC10 = OffsetScaling(
        offset_inputs={i: x_offset7[inputs7[i]] for i in range(len(inputs7))},
        factor_inputs={i: x_factor7[inputs7[i]] for i in range(len(inputs7))},
        offset_outputs={i: y_offset_cooler10[cooler10[i]] for i in range(len(cooler10))},
        factor_outputs={i: y_factor_cooler10[cooler10[i]] for i in range(len(cooler10))}
    )

scalerC11 = OffsetScaling(
        offset_inputs={i: x_offset7[inputs7[i]] for i in range(len(inputs7))},
        factor_inputs={i: x_factor7[inputs7[i]] for i in range(len(inputs7))},
        offset_outputs={i: y_offset_cooler11[cooler11[i]] for i in range(len(cooler11))},
        factor_outputs={i: y_factor_cooler11[cooler11[i]] for i in range(len(cooler11))}
    )

scalerT6 = OffsetScaling(
        offset_inputs={i: x_offset8[inputs8[i]] for i in range(len(inputs8))},
        factor_inputs={i: x_factor8[inputs8[i]] for i in range(len(inputs8))},
        offset_outputs={i: y_offsetT[outputT[i]] for i in range(len(outputT))},
        factor_outputs={i: y_factorT[outputT[i]] for i in range(len(outputT))}
    )

scalerE6 = OffsetScaling(
        offset_inputs={i: x_offset8[inputs8[i]] for i in range(len(inputs8))},
        factor_inputs={i: x_factor8[inputs8[i]] for i in range(len(inputs8))},
        offset_outputs={i: y_offsetE[outputE[i]] for i in range(len(outputE))},
        factor_outputs={i: y_factorE[outputE[i]] for i in range(len(outputE))}
    )

#%%
scalerV3 = OffsetScaling(
        offset_inputs={i: x_offset9[inputs9[i]] for i in range(len(inputs9))},
        factor_inputs={i: x_factor9[inputs9[i]] for i in range(len(inputs9))},
        offset_outputs={i: y_offset_valve3[valve3[i]] for i in range(len(valve3))},
        factor_outputs={i: y_factor_valve3[valve3[i]] for i in range(len(valve3))}
    )

scalerV5 = OffsetScaling(
        offset_inputs={i: x_offset9[inputs9[i]] for i in range(len(inputs9))},
        factor_inputs={i: x_factor9[inputs9[i]] for i in range(len(inputs9))},
        offset_outputs={i: y_offset_valve5[valve5[i]] for i in range(len(valve5))},
        factor_outputs={i: y_factor_valve5[valve5[i]] for i in range(len(valve5))}
    )

scalerV6 = OffsetScaling(
        offset_inputs={i: x_offset9[inputs9[i]] for i in range(len(inputs9))},
        factor_inputs={i: x_factor9[inputs9[i]] for i in range(len(inputs9))},
        offset_outputs={i: y_offset_valve6[valve6[i]] for i in range(len(valve6))},
        factor_outputs={i: y_factor_valve6[valve6[i]] for i in range(len(valve6))}
    )

scalerV8 = OffsetScaling(
        offset_inputs={i: x_offset9[inputs9[i]] for i in range(len(inputs9))},
        factor_inputs={i: x_factor9[inputs9[i]] for i in range(len(inputs9))},
        offset_outputs={i: y_offset_valve8[valve8[i]] for i in range(len(valve8))},
        factor_outputs={i: y_factor_valve8[valve8[i]] for i in range(len(valve8))}
    )

scalerV9 = OffsetScaling(
        offset_inputs={i: x_offset9[inputs9[i]] for i in range(len(inputs9))},
        factor_inputs={i: x_factor9[inputs9[i]] for i in range(len(inputs9))},
        offset_outputs={i: y_offset_valve9[valve9[i]] for i in range(len(valve9))},
        factor_outputs={i: y_factor_valve9[valve9[i]] for i in range(len(valve9))}
    )

scalerV10 = OffsetScaling(
        offset_inputs={i: x_offset9[inputs9[i]] for i in range(len(inputs9))},
        factor_inputs={i: x_factor9[inputs9[i]] for i in range(len(inputs9))},
        offset_outputs={i: y_offset_valve10[valve10[i]] for i in range(len(valve10))},
        factor_outputs={i: y_factor_valve10[valve10[i]] for i in range(len(valve10))}
    )

scalerH1 = OffsetScaling(
        offset_inputs={i: x_offset10[inputs10[i]] for i in range(len(inputs10))},
        factor_inputs={i: x_factor10[inputs10[i]] for i in range(len(inputs10))},
        offset_outputs={i: y_offsetE[outputE[i]] for i in range(len(outputE))},
        factor_outputs={i: y_factorE[outputE[i]] for i in range(len(outputE))}
    )

scalerD1 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis1[dis1[i]] for i in range(len(dis1))},
        factor_outputs={i: y_factor_dis1[dis1[i]] for i in range(len(dis1))}
    )

scalerD2 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis2[dis2[i]] for i in range(len(dis2))},
        factor_outputs={i: y_factor_dis2[dis2[i]] for i in range(len(dis2))}
    )

scalerD3 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis3[dis3[i]] for i in range(len(dis3))},
        factor_outputs={i: y_factor_dis3[dis3[i]] for i in range(len(dis3))}
    )

scalerD4 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis4[dis4[i]] for i in range(len(dis4))},
        factor_outputs={i: y_factor_dis4[dis4[i]] for i in range(len(dis4))}
    )

scalerD5 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis5[dis5[i]] for i in range(len(dis5))},
        factor_outputs={i: y_factor_dis5[dis5[i]] for i in range(len(dis5))}
    )

scalerD6 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis6[dis6[i]] for i in range(len(dis6))},
        factor_outputs={i: y_factor_dis6[dis6[i]] for i in range(len(dis6))}
    )

scalerD7 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis7[dis7[i]] for i in range(len(dis7))},
        factor_outputs={i: y_factor_dis7[dis7[i]] for i in range(len(dis7))}
    )

scalerD8 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis8[dis8[i]] for i in range(len(dis8))},
        factor_outputs={i: y_factor_dis8[dis8[i]] for i in range(len(dis8))}
    )

scalerD9 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis9[dis9[i]] for i in range(len(dis9))},
        factor_outputs={i: y_factor_dis9[dis9[i]] for i in range(len(dis9))}
    )

scalerD10 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis10[dis10[i]] for i in range(len(dis10))},
        factor_outputs={i: y_factor_dis10[dis10[i]] for i in range(len(dis10))}
    )

scalerD11 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis11[dis11[i]] for i in range(len(dis11))},
        factor_outputs={i: y_factor_dis11[dis11[i]] for i in range(len(dis11))}
    )

scalerD12 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis12[dis12[i]] for i in range(len(dis12))},
        factor_outputs={i: y_factor_dis12[dis12[i]] for i in range(len(dis12))}
    )

scalerD13 = OffsetScaling(
        offset_inputs={i: x_offset11[inputs11[i]] for i in range(len(inputs11))},
        factor_inputs={i: x_factor11[inputs11[i]] for i in range(len(inputs11))},
        offset_outputs={i: y_offset_dis13[dis13[i]] for i in range(len(dis13))},
        factor_outputs={i: y_factor_dis13[dis13[i]] for i in range(len(dis13))}
    )

scalerC01 = OffsetScaling(
        offset_inputs={i: x_offset12[inputs12[i]] for i in range(len(inputs12))},
        factor_inputs={i: x_factor12[inputs12[i]] for i in range(len(inputs12))},
        offset_outputs={i: y_offset_cooler1_2[cooler1_2[i]] for i in range(len(cooler1_2))},
        factor_outputs={i: y_factor_cooler1_2[cooler1_2[i]] for i in range(len(cooler1_2))}
    )

scalerC02 = OffsetScaling(
        offset_inputs={i: x_offset12[inputs12[i]] for i in range(len(inputs12))},
        factor_inputs={i: x_factor12[inputs12[i]] for i in range(len(inputs12))},
        offset_outputs={i: y_offset_cooler2_2[cooler2_2[i]] for i in range(len(cooler2_2))},
        factor_outputs={i: y_factor_cooler2_2[cooler2_2[i]] for i in range(len(cooler2_2))}
    )

scalerC03 = OffsetScaling(
        offset_inputs={i: x_offset12[inputs12[i]] for i in range(len(inputs12))},
        factor_inputs={i: x_factor12[inputs12[i]] for i in range(len(inputs12))},
        offset_outputs={i: y_offset_cooler3_2[cooler3_2[i]] for i in range(len(cooler3_2))},
        factor_outputs={i: y_factor_cooler3_2[cooler3_2[i]] for i in range(len(cooler3_2))}
    )

scalerC04 = OffsetScaling(
        offset_inputs={i: x_offset12[inputs12[i]] for i in range(len(inputs12))},
        factor_inputs={i: x_factor12[inputs12[i]] for i in range(len(inputs12))},
        offset_outputs={i: y_offset_cooler4_2[cooler4_2[i]] for i in range(len(cooler4_2))},
        factor_outputs={i: y_factor_cooler4_2[cooler4_2[i]] for i in range(len(cooler4_2))}
    )

scalerC05 = OffsetScaling(
        offset_inputs={i: x_offset12[inputs12[i]] for i in range(len(inputs12))},
        factor_inputs={i: x_factor12[inputs12[i]] for i in range(len(inputs12))},
        offset_outputs={i: y_offset_cooler5_2[cooler5_2[i]] for i in range(len(cooler5_2))},
        factor_outputs={i: y_factor_cooler5_2[cooler5_2[i]] for i in range(len(cooler5_2))}
    )

scalerC06 = OffsetScaling(
        offset_inputs={i: x_offset12[inputs12[i]] for i in range(len(inputs12))},
        factor_inputs={i: x_factor12[inputs12[i]] for i in range(len(inputs12))},
        offset_outputs={i: y_offset_cooler6_2[cooler6_2[i]] for i in range(len(cooler6_2))},
        factor_outputs={i: y_factor_cooler6_2[cooler6_2[i]] for i in range(len(cooler6_2))}
    )

scalerC07 = OffsetScaling(
        offset_inputs={i: x_offset12[inputs12[i]] for i in range(len(inputs12))},
        factor_inputs={i: x_factor12[inputs12[i]] for i in range(len(inputs12))},
        offset_outputs={i: y_offset_cooler7_2[cooler7_2[i]] for i in range(len(cooler7_2))},
        factor_outputs={i: y_factor_cooler7_2[cooler7_2[i]] for i in range(len(cooler7_2))}
    )


#%%
scaled_lb=np.array([0,0,0,0])
scaled_ub=np.array([1,1,1,1])
scaled_lb2=np.array([0,0,0,0,0])
scaled_ub2=np.array([1,1,1,1,1])
scaled_lb3=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
scaled_ub3=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])
scaled_lb4=np.array([0,0,0,0,0,0,0,0])
scaled_ub4=np.array([1,1,1,1,1,1,1,1])
scaled_lb5=np.array([0,0,0,0,0,0,0])
scaled_ub5=np.array([1,1,1,1,1,1,1])
#%%
scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}
scaled_input_bounds2 = {i: (scaled_lb2[i], scaled_ub2[i]) for i in range(len(inputs4))}
scaled_input_bounds3 = {i: (scaled_lb3[i], scaled_ub3[i]) for i in range(len(inputs5))}
scaled_input_bounds4 = {i: (scaled_lb4[i], scaled_ub4[i]) for i in range(len(inputs6))}
scaled_input_bounds5 = {i: (scaled_lb5[i], scaled_ub5[i]) for i in range(len(inputs11))}
#%%
# create a network definition from the Keras model 
netT = load_keras_sequential(nn_compressor_T, scaling_object=scalerT, scaled_input_bounds=scaled_input_bounds)
m.compressor_T1.build_formulation(FullSpaceSmoothNNFormulation(netT))
netT2= load_keras_sequential(nn_compressor_T, scaling_object=scalerT, scaled_input_bounds=scaled_input_bounds)
m.compressor_T2.build_formulation(FullSpaceSmoothNNFormulation(netT2))
netE = load_keras_sequential(nn_compressor_E, scaling_object=scalerE, scaled_input_bounds=scaled_input_bounds)
formulation3_mixed = FullSpaceNNFormulation(netE,activation_constraints={"relu": ComplementarityReLUActivation()})
m.compressor_E1.build_formulation(formulation3_mixed)
netE2 = load_keras_sequential(nn_compressor_E, scaling_object=scalerE, scaled_input_bounds=scaled_input_bounds)
formulation3_mixed2 = FullSpaceNNFormulation(netE2,activation_constraints={"relu": ComplementarityReLUActivation()})
m.compressor_E2.build_formulation(formulation3_mixed2)
netT3 = load_keras_sequential(nn_compressor_T_HP, scaling_object=scalerT2, scaled_input_bounds=scaled_input_bounds)
m.compressor_T3.build_formulation(FullSpaceSmoothNNFormulation(netT3))
netT4 = load_keras_sequential(nn_compressor_T_HP, scaling_object=scalerT2, scaled_input_bounds=scaled_input_bounds)
m.compressor_T4.build_formulation(FullSpaceSmoothNNFormulation(netT4))
netE3 = load_keras_sequential(nn_compressor_E_HP, scaling_object=scalerE2, scaled_input_bounds=scaled_input_bounds)
formulation3_mixed3 = FullSpaceNNFormulation(netE3,activation_constraints={"relu": ComplementarityReLUActivation()})
m.compressor_E3.build_formulation(formulation3_mixed3)
netE4 = load_keras_sequential(nn_compressor_E_HP, scaling_object=scalerE2, scaled_input_bounds=scaled_input_bounds)
formulation3_mixed4 = FullSpaceNNFormulation(netE4,activation_constraints={"relu": ComplementarityReLUActivation()})
m.compressor_E4.build_formulation(formulation3_mixed4)

netT5 = load_keras_sequential(nn_compressor_T_HP2, scaling_object=scalerT3, scaled_input_bounds=scaled_input_bounds)
m.compressor_H2_T.build_formulation(FullSpaceSmoothNNFormulation(netT5))
netE5 = load_keras_sequential(nn_compressor_E_HP2, scaling_object=scalerE3, scaled_input_bounds=scaled_input_bounds)
formulation3_mixed5 = FullSpaceNNFormulation(netE5,activation_constraints={"relu": ComplementarityReLUActivation()})
m.compressor_H2_E.build_formulation(formulation3_mixed5)

netT6 = load_keras_sequential(nn_mixer_feed, scaling_object=scalerT4, scaled_input_bounds=scaled_input_bounds2)
m.mixer_feed.build_formulation(FullSpaceSmoothNNFormulation(netT6))

netT7 = load_keras_sequential(nn_mixer_reactor, scaling_object=scalerT5, scaled_input_bounds=scaled_input_bounds3)
formulation3_mixed6 = FullSpaceNNFormulation(netT7,activation_constraints={"relu": ComplementarityReLUActivation()})
m.mixer_reactor.build_formulation(formulation3_mixed6)

netR1 = load_keras_sequential(nn_reactor_comp, scaling_object=scalerR1, scaled_input_bounds=scaled_input_bounds4)
m.reactor_comp.build_formulation(FullSpaceSmoothNNFormulation(netR1))
netR2 = load_keras_sequential(nn_reactor_ch3oh, scaling_object=scalerR2, scaled_input_bounds=scaled_input_bounds4)
m.reactor_ch3oh.build_formulation(FullSpaceSmoothNNFormulation(netR2))
netR3 = load_keras_sequential(nn_reactor_h2, scaling_object=scalerR3, scaled_input_bounds=scaled_input_bounds4)
m.reactor_h2.build_formulation(FullSpaceSmoothNNFormulation(netR3))
netR4 = load_keras_sequential(nn_reactor_T, scaling_object=scalerR4, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed7 = FullSpaceNNFormulation(netR4,activation_constraints={"relu": ComplementarityReLUActivation()})
m.reactor_T.build_formulation(formulation3_mixed7)
netR5 = load_keras_sequential(nn_reactor_P, scaling_object=scalerR5, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed8 = FullSpaceNNFormulation(netR5,activation_constraints={"relu": ComplementarityReLUActivation()})
m.reactor_P.build_formulation(formulation3_mixed8)

netC1 = load_keras_sequential(nn_cooler_vap_co2_co, scaling_object=scalerC1, scaled_input_bounds=scaled_input_bounds4)
m.cooler_vap_co2_co.build_formulation(ReluBigMFormulation(netC1))
netC3 = load_keras_sequential(nn_cooler_vap_h2, scaling_object=scalerC3, scaled_input_bounds=scaled_input_bounds4)
m.cooler_vap_h2.build_formulation(FullSpaceSmoothNNFormulation(netC3))
netC5 = load_keras_sequential(nn_cooler_liq_ch3oh_h2o, scaling_object=scalerC5, scaled_input_bounds=scaled_input_bounds4)
m.cooler_liq_ch3oh_h2o.build_formulation(FullSpaceSmoothNNFormulation(netC5))
netC9 = load_keras_sequential(nn_cooler_E, scaling_object=scalerC9, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed9 = FullSpaceNNFormulation(netC9,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_power.build_formulation(formulation3_mixed9)
netC10 = load_keras_sequential(nn_cooler_VD, scaling_object=scalerC10, scaled_input_bounds=scaled_input_bounds4)
m.cooler_VD.build_formulation(FullSpaceSmoothNNFormulation(netC10))
netC11 = load_keras_sequential(nn_cooler_LD, scaling_object=scalerC11, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed10 = FullSpaceNNFormulation(netC11,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_LD.build_formulation(formulation3_mixed10)

netT8 = load_keras_sequential(nn_compressor_T_mix, scaling_object=scalerT6, scaled_input_bounds=scaled_input_bounds4)
m.compressor_Mix_T.build_formulation(FullSpaceSmoothNNFormulation(netT8))
netE8 = load_keras_sequential(nn_compressor_E_mix, scaling_object=scalerE6, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed11 = FullSpaceNNFormulation(netE8,activation_constraints={"relu": ComplementarityReLUActivation()})
m.compressor_Mix_E.build_formulation(formulation3_mixed11)

netV3 = load_keras_sequential(nn_valve_vap_co, scaling_object=scalerV3, scaled_input_bounds=scaled_input_bounds4)
m.valve_vap_co.build_formulation(FullSpaceSmoothNNFormulation(netV3))
netV5 = load_keras_sequential(nn_valve_liq_ch3oh_h2o, scaling_object=scalerV5, scaled_input_bounds=scaled_input_bounds4)
m.valve_liq_ch3oh_h2o.build_formulation(FullSpaceSmoothNNFormulation(netV5))
netV6 = load_keras_sequential(nn_valve_liq_co2, scaling_object=scalerV6, scaled_input_bounds=scaled_input_bounds4)
m.valve_liq_co2.build_formulation(ReluBigMFormulation(netV6))
netV8 = load_keras_sequential(nn_valve_T, scaling_object=scalerV8, scaled_input_bounds=scaled_input_bounds4)
m.valve_T.build_formulation(ReluBigMFormulation(netV8))
netV9 = load_keras_sequential(nn_valve_VD, scaling_object=scalerV9, scaled_input_bounds=scaled_input_bounds4)
m.valve_VD.build_formulation(FullSpaceNNFormulation(netV9))
netV10 = load_keras_sequential(nn_valve_LD, scaling_object=scalerV10, scaled_input_bounds=scaled_input_bounds4)
m.valve_LD.build_formulation(ReluBigMFormulation(netV10))

netH1 = load_keras_sequential(nn_heater, scaling_object=scalerH1, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed12 = FullSpaceNNFormulation(netH1,activation_constraints={"relu": ComplementarityReLUActivation()})
m.heater.build_formulation(formulation3_mixed12)

netD1 = load_keras_sequential(nn_dis_vap_co2, scaling_object=scalerD1, scaled_input_bounds=scaled_input_bounds5)
m.dis_vap_co2.build_formulation(ReluBigMFormulation(netD1))
netD2 = load_keras_sequential(nn_dis_vap_ch3oh, scaling_object=scalerD2, scaled_input_bounds=scaled_input_bounds5)
m.dis_vap_ch3oh.build_formulation(FullSpaceNNFormulation(netD2))
netD3 = load_keras_sequential(nn_dis_vap_co, scaling_object=scalerD3, scaled_input_bounds=scaled_input_bounds5)
m.dis_vap_co.build_formulation(ReluBigMFormulation(netD3))
netD4 = load_keras_sequential(nn_dis_vap_h2o, scaling_object=scalerD4, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed13 = FullSpaceNNFormulation(netD4,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_vap_h2o.build_formulation(formulation3_mixed13)
netD5 = load_keras_sequential(nn_dis_con_T, scaling_object=scalerD5, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed14 = FullSpaceNNFormulation(netD5,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_con_T.build_formulation(formulation3_mixed14)
netD6 = load_keras_sequential(nn_dis_reb_T, scaling_object=scalerD6, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed15 = FullSpaceNNFormulation(netD6,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_reb_T.build_formulation(formulation3_mixed15)
netD7 = load_keras_sequential(nn_dis_con_duty, scaling_object=scalerD7, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed16 = FullSpaceNNFormulation(netD7,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_con_duty.build_formulation(formulation3_mixed16)
netD8 = load_keras_sequential(nn_dis_reb_duty, scaling_object=scalerD8, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed17 = FullSpaceNNFormulation(netD8,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_reb_duty.build_formulation(formulation3_mixed17)
netD9 = load_keras_sequential(nn_dis_VD, scaling_object=scalerD9, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed18 = FullSpaceNNFormulation(netD9,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_VD.build_formulation(formulation3_mixed18)
netD10 = load_keras_sequential(nn_dis_LD, scaling_object=scalerD10, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed19 = FullSpaceNNFormulation(netD10,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_LD.build_formulation(formulation3_mixed19)
netD11 = load_keras_sequential(nn_dis_max_vap, scaling_object=scalerD11, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed20 = FullSpaceNNFormulation(netD11,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_max_vap.build_formulation(formulation3_mixed20)
netD12 = load_keras_sequential(nn_dis_no_stage, scaling_object=scalerD12, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed21 = FullSpaceNNFormulation(netD12,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_no_stage.build_formulation(formulation3_mixed21)
netD13 = load_keras_sequential(nn_dis_feed_stage, scaling_object=scalerD13, scaled_input_bounds=scaled_input_bounds5)
formulation3_mixed22 = FullSpaceNNFormulation(netD13,activation_constraints={"relu": ComplementarityReLUActivation()})
m.dis_feed_stage.build_formulation(formulation3_mixed22)

netC01 = load_keras_sequential(nn_cooler_liq_co2_2, scaling_object=scalerC01, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed23 = FullSpaceNNFormulation(netC01,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_liq_co2_2.build_formulation(formulation3_mixed23)
netC02 = load_keras_sequential(nn_cooler_liq_ch3oh_2, scaling_object=scalerC02, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed24 = FullSpaceNNFormulation(netC02,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_liq_ch3oh_2.build_formulation(formulation3_mixed24)
netC03 = load_keras_sequential(nn_cooler_liq_co_2, scaling_object=scalerC03, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed25 = FullSpaceNNFormulation(netC03,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_liq_co_2.build_formulation(formulation3_mixed25)
netC04 = load_keras_sequential(nn_cooler_liq_h2o_2, scaling_object=scalerC04, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed26 = FullSpaceNNFormulation(netC04,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_liq_h2o_2.build_formulation(formulation3_mixed26)
netC05 = load_keras_sequential(nn_cooler_power_2, scaling_object=scalerC05, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed27 = FullSpaceNNFormulation(netC05,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_power_2.build_formulation(formulation3_mixed27)
netC06 = load_keras_sequential(nn_cooler_VD_2, scaling_object=scalerC06, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed28 = FullSpaceNNFormulation(netC06,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_VD_2.build_formulation(formulation3_mixed28)
netC07 = load_keras_sequential(nn_cooler_LD_2, scaling_object=scalerC07, scaled_input_bounds=scaled_input_bounds4)
formulation3_mixed29 = FullSpaceNNFormulation(netC07,activation_constraints={"relu": ComplementarityReLUActivation()})
m.cooler_LD_2.build_formulation(formulation3_mixed29)

#%%
"""
m.FV1= pyo.Var(within=pyo.NonNegativeReals,initialize=150)
m.FV2= pyo.Var(within=pyo.NonNegativeReals,initialize=1)

m.compc1=pyo.Var(within=pyo.NonNegativeReals,initialize=500000)
m.compc2=pyo.Var(within=pyo.NonNegativeReals,initialize=500000)
m.compc3=pyo.Var(within=pyo.NonNegativeReals,initialize=500000)
m.compc4=pyo.Var(within=pyo.NonNegativeReals,initialize=500000)
m.compc5=pyo.Var(within=pyo.NonNegativeReals,initialize=500000)
m.compc6=pyo.Var(within=pyo.NonNegativeReals,initialize=500000)

m.reactorc=pyo.Var(within=pyo.NonNegativeReals,initialize=500000)

m.v_max1=pyo.Var(within=pyo.NonNegativeReals,initialize=1)
m.V_flow1=pyo.Var(within=pyo.NonNegativeReals,initialize=10)
m.area1=pyo.Var(within=pyo.NonNegativeReals,initialize=100)
m.D1=pyo.Var(within=pyo.NonNegativeReals,initialize=1)
m.t1=pyo.Var(within=pyo.NonNegativeReals,initialize=0.1)
m.W1=pyo.Var(within=pyo.NonNegativeReals,initialize=1000)
m.vesselc1=pyo.Var(within=pyo.NonNegativeReals,initialize=50000)

m.v_max2=pyo.Var(within=pyo.NonNegativeReals,initialize=1)
m.V_flow2=pyo.Var(within=pyo.NonNegativeReals,initialize=10)
m.area2=pyo.Var(within=pyo.NonNegativeReals,initialize=100)
m.D2=pyo.Var(within=pyo.NonNegativeReals,initialize=10)
m.t2=pyo.Var(within=pyo.NonNegativeReals,initialize=0.1)
m.W2=pyo.Var(within=pyo.NonNegativeReals,initialize=1000)
m.vesselc2=pyo.Var(within=pyo.NonNegativeReals,initialize=50000)

m.FV3= pyo.Var(within=pyo.NonNegativeReals,initialize=50)
m.v_max3=pyo.Var(within=pyo.NonNegativeReals,initialize=100)
m.V_flow3=pyo.Var(within=pyo.NonNegativeReals,initialize=10)
m.area3=pyo.Var(within=pyo.NonNegativeReals,initialize=1)
m.D3=pyo.Var(within=pyo.NonNegativeReals,initialize=10)
m.t3=pyo.Var(within=pyo.NonNegativeReals,initialize=0.1)
m.W3=pyo.Var(within=pyo.NonNegativeReals,initialize=1000)
m.vesselc3=pyo.Var(within=pyo.NonNegativeReals,initialize=50000)

m.FV4= pyo.Var(within=pyo.NonNegativeReals,initialize=15)
m.v_max4=pyo.Var(within=pyo.NonNegativeReals,initialize=1)
m.V_flow4=pyo.Var(within=pyo.NonNegativeReals,initialize=10)
m.area4=pyo.Var(within=pyo.NonNegativeReals,initialize=1)
m.D4=pyo.Var(within=pyo.NonNegativeReals,initialize=10)
m.t4=pyo.Var(within=pyo.NonNegativeReals,initialize=0.1)
m.W4=pyo.Var(within=pyo.NonNegativeReals,initialize=1000)
m.vesselc4=pyo.Var(within=pyo.NonNegativeReals,initialize=50000)

m.total_capex = pyo.Var(within=pyo.NonNegativeReals,initialize=1)
"""
#%%
"""
#1
m.total_opex = pyo.Var(bounds=(1e-20,20),initialize=10)
m.total_capex = pyo.Var(bounds=(1e-20,100),initialize=35)

m.compc1=pyo.Var(bounds=(1e-20,200),initialize=100)
m.compc2=pyo.Var(bounds=(1e-20,200),initialize=100)
m.compc3=pyo.Var(bounds=(1e-20,200),initialize=100)
m.compc4=pyo.Var(bounds=(1e-20,200),initialize=100)
m.compc5=pyo.Var(bounds=(1e-20,200),initialize=150)
m.compc6=pyo.Var(bounds=(1e-20,400),initialize=200)

m.reactorc=pyo.Var(bounds=(1e-20,100),initialize=50)

m.vesselc4=pyo.Var(bounds=(1e-20,100),initialize=50)

m.TAC=pyo.Var(bounds=(1e-20,50),initialize=1)
m.Total_Energy=pyo.Var(bounds=(1e-20,1000000), initialize=50000)
m.cooler_liq_ch3oh_2_out=pyo.Var(bounds=(1e-20,3000), initialize=3000)
 
#2
m.total_opex = pyo.Var(bounds=(1e-20,20),initialize=7.124873429997057)
m.total_capex = pyo.Var(bounds=(1e-20,100),initialize=12.94993918782061)

m.compc1=pyo.Var(bounds=(1e-20,200),initialize=63.372261224934434)
m.compc2=pyo.Var(bounds=(1e-20,200),initialize=66.48925184208262)
m.compc3=pyo.Var(bounds=(1e-20,200),initialize=59.73560441893049)
m.compc4=pyo.Var(bounds=(1e-20,200),initialize=58.704573052654716)
m.compc5=pyo.Var(bounds=(1e-20,200),initialize=113.59475424495439)
m.compc6=pyo.Var(bounds=(1e-20,400),initialize=157.24613752942255)

m.reactorc=pyo.Var(bounds=(1e-20,100),initialize=50.65425422396129)

m.vesselc4=pyo.Var(bounds=(1e-20,100),initialize=51.80024447844887)

m.TAC=pyo.Var(bounds=(1e-20,50),initialize=0.47701909618625543)
m.Total_Energy=pyo.Var(bounds=(1e-20,1000000), initialize=24861.278824516696)
m.cooler_liq_ch3oh_2_out=pyo.Var(bounds=(1e-20,3000), initialize=1710.8684934959192)
"""

#3
m.total_opex = pyo.Var(bounds=(2,8),initialize=7.124873429997057)
m.total_capex = pyo.Var(bounds=(8,14),initialize=12.94993918782061)

m.compc1=pyo.Var(bounds=(55,70),initialize=63.372261224934434)
m.compc2=pyo.Var(bounds=(55,70),initialize=66.48925184208262)
m.compc3=pyo.Var(bounds=(50,65),initialize=59.73560441893049)
m.compc4=pyo.Var(bounds=(50,65),initialize=58.704573052654716)
m.compc5=pyo.Var(bounds=(100,115),initialize=113.59475424495439)
m.compc6=pyo.Var(bounds=(125,150),initialize=157.24613752942255)

m.reactorc=pyo.Var(bounds=(40,55),initialize=50.65425422396129)

m.vesselc4=pyo.Var(bounds=(40,60),initialize=51.80024447844887)


m.TAC=pyo.Var(bounds=(0,2),initialize=0.47701909618625543)
m.Total_Energy=pyo.Var(bounds=(15000,30000), initialize=24861.278824516696)
m.cooler_liq_ch3oh_2_out=pyo.Var(bounds=(1000,2000), initialize=1710.8684934959192)


#%%
m.obj = pyo.Objective(expr=m.TAC,sense=pyo.minimize) 

#%%
# compressor1: definition
m.con1 = pyo.Constraint(expr=m.compressor_T1.inputs[0] == m.compressor_E1.inputs[0])
m.con2 = pyo.Constraint(expr=m.compressor_T1.inputs[1] == m.compressor_E1.inputs[1])
m.con3 = pyo.Constraint(expr=m.compressor_T1.inputs[2] == m.compressor_E1.inputs[2])
m.con4 = pyo.Constraint(expr=m.compressor_T1.inputs[3] == m.compressor_E1.inputs[3])

# compressor2: definition
m.con5 = pyo.Constraint(expr=m.compressor_T2.inputs[0] == m.compressor_E2.inputs[0])
m.con6 = pyo.Constraint(expr=m.compressor_T2.inputs[1] == m.compressor_E2.inputs[1])
m.con7 = pyo.Constraint(expr=m.compressor_T2.inputs[2] == m.compressor_E2.inputs[2])
m.con8 = pyo.Constraint(expr=m.compressor_T2.inputs[3] == m.compressor_E2.inputs[3])

# compressor3: definition
m.con9 = pyo.Constraint(expr=m.compressor_T3.inputs[0] == m.compressor_E3.inputs[0])
m.con10 = pyo.Constraint(expr=m.compressor_T3.inputs[1] == m.compressor_E3.inputs[1])
m.con11 = pyo.Constraint(expr=m.compressor_T3.inputs[2] == m.compressor_E3.inputs[2])
m.con12 = pyo.Constraint(expr=m.compressor_T3.inputs[3] == m.compressor_E3.inputs[3])

# compressor4: definition
m.con13 = pyo.Constraint(expr=m.compressor_T4.inputs[0] == m.compressor_E4.inputs[0])
m.con14 = pyo.Constraint(expr=m.compressor_T4.inputs[1] == m.compressor_E4.inputs[1])
m.con15 = pyo.Constraint(expr=m.compressor_T4.inputs[2] == m.compressor_E4.inputs[2])
m.con16 = pyo.Constraint(expr=m.compressor_T4.inputs[3] == m.compressor_E4.inputs[3])

# compressor5_for_H2: definition
m.con17 = pyo.Constraint(expr=m.compressor_H2_T.inputs[0] == m.compressor_H2_E.inputs[0])
m.con18 = pyo.Constraint(expr=m.compressor_H2_T.inputs[1] == m.compressor_H2_E.inputs[1])
m.con19 = pyo.Constraint(expr=m.compressor_H2_T.inputs[2] == m.compressor_H2_E.inputs[2])
m.con20 = pyo.Constraint(expr=m.compressor_H2_T.inputs[3] == m.compressor_H2_E.inputs[3])

#linking compressors chain
#equating flowrates of 1 to flowrates of 2
m.con141 = pyo.Constraint(expr=m.compressor_T1.inputs[0] == m.compressor_T2.inputs[0])
m.con142 = pyo.Constraint(expr=m.compressor_T2.inputs[0] == m.compressor_T3.inputs[0])
m.con143 = pyo.Constraint(expr=m.compressor_T3.inputs[0] == m.compressor_T4.inputs[0])

#equating input pressure of 2 to output pressure of 1
m.con144 = pyo.Constraint(expr=m.compressor_T2.inputs[2] == m.compressor_T1.inputs[2]*m.compressor_T1.inputs[3])
m.con145 = pyo.Constraint(expr=m.compressor_T3.inputs[2] == m.compressor_T2.inputs[2]*m.compressor_T2.inputs[3])
m.con146 = pyo.Constraint(expr=m.compressor_T4.inputs[2] == m.compressor_T3.inputs[2]*m.compressor_T3.inputs[3])

#input to CO2 chain
m.con164 = pyo.Constraint(expr=m.compressor_T1.inputs[0] == 2000)
m.con165 = pyo.Constraint(expr=m.compressor_T1.inputs[1] == 25)
m.con166 = pyo.Constraint(expr=m.compressor_T1.inputs[2] == 100)

#input to h2
m.con167 = pyo.Constraint(expr=m.compressor_H2_T.inputs[0] >= 4000)
m.con1671 = pyo.Constraint(expr=m.compressor_H2_T.inputs[0] <= 8000) 
m.con168 = pyo.Constraint(expr=m.compressor_H2_T.inputs[1] == 25)
m.con169 = pyo.Constraint(expr=m.compressor_H2_T.inputs[2] == 3000)

#%%
#linking CO2 compressed feed output to mixer input 
m.con147 = pyo.Constraint(expr=m.compressor_T4.inputs[0] == m.mixer_feed.inputs[0])
m.con148 = pyo.Constraint(expr=m.compressor_T4.outputs[0] == m.mixer_feed.inputs[1])
#linking H2 compressed feed output to mixer input 
m.con149 = pyo.Constraint(expr=m.compressor_H2_T.inputs[0] == m.mixer_feed.inputs[2])
m.con150 = pyo.Constraint(expr=m.compressor_H2_T.outputs[0] == m.mixer_feed.inputs[3])
m.con151 = pyo.Constraint(expr=m.compressor_H2_T.inputs[2]*m.compressor_H2_T.inputs[3] == m.mixer_feed.inputs[4])

#%%
#defining second mixer input1
m.con152 = pyo.Constraint(expr=m.mixer_reactor.inputs[0]==m.mixer_feed.inputs[0])
m.con153 = pyo.Constraint(expr=m.mixer_reactor.inputs[1]==m.mixer_feed.inputs[2])
m.con154 = pyo.Constraint(expr=m.mixer_reactor.inputs[2]==0)
m.con155 = pyo.Constraint(expr=m.mixer_reactor.inputs[3]==0)
m.con156 = pyo.Constraint(expr=m.mixer_reactor.inputs[4]==0)
m.con157 = pyo.Constraint(expr=m.mixer_reactor.inputs[5]==m.mixer_feed.outputs[0])
#defining second mixer input2 
m.con158 = pyo.Constraint(expr=m.mixer_reactor.inputs[6]<=1.01*m.compressor_Mix_T.inputs[0])
m.con159 = pyo.Constraint(expr=m.mixer_reactor.inputs[7]<=1.01*m.compressor_Mix_T.inputs[1])
m.con160 = pyo.Constraint(expr=m.mixer_reactor.inputs[8]<=1.01*m.compressor_Mix_T.inputs[2])
m.con161 = pyo.Constraint(expr=m.mixer_reactor.inputs[9]<=1.01*m.compressor_Mix_T.inputs[3])
m.con162 = pyo.Constraint(expr=m.mixer_reactor.inputs[10]<=1.01*m.compressor_Mix_T.inputs[4])
m.con163 = pyo.Constraint(expr=m.mixer_reactor.inputs[11]<=1.01*m.compressor_Mix_T.outputs[0])

m.con1582 = pyo.Constraint(expr=m.mixer_reactor.inputs[6]>=0.99*m.compressor_Mix_T.inputs[0])
m.con1592 = pyo.Constraint(expr=m.mixer_reactor.inputs[7]>=0.99*m.compressor_Mix_T.inputs[1])
m.con1602 = pyo.Constraint(expr=m.mixer_reactor.inputs[8]>=0.99*m.compressor_Mix_T.inputs[2])
m.con1612 = pyo.Constraint(expr=m.mixer_reactor.inputs[9]>=0.99*m.compressor_Mix_T.inputs[3])
m.con1622 = pyo.Constraint(expr=m.mixer_reactor.inputs[10]>=0.99*m.compressor_Mix_T.inputs[4])
m.con1632 = pyo.Constraint(expr=m.mixer_reactor.inputs[11]>=0.99*m.compressor_Mix_T.outputs[0])

m.con170 = pyo.Constraint(expr=m.compressor_T4.inputs[2]*m.compressor_T4.inputs[3]==m.compressor_H2_T.inputs[2]*m.compressor_H2_T.inputs[3])
m.con171 = pyo.Constraint(expr=m.mixer_feed.inputs[4]>=7800)
m.con1712 = pyo.Constraint(expr=m.mixer_feed.inputs[4]<=10000)
m.con172 = pyo.Constraint(expr=m.mixer_reactor.inputs[12]==m.mixer_feed.inputs[4])

#%%
# reactor inlet equate
m.con21 = pyo.Constraint(expr=m.reactor_comp.inputs[0] == m.reactor_ch3oh.inputs[0])
m.con22 = pyo.Constraint(expr=m.reactor_comp.inputs[1] == m.reactor_ch3oh.inputs[1])
m.con23 = pyo.Constraint(expr=m.reactor_comp.inputs[2] == m.reactor_ch3oh.inputs[2])
m.con24 = pyo.Constraint(expr=m.reactor_comp.inputs[3] == m.reactor_ch3oh.inputs[3])
m.con25 = pyo.Constraint(expr=m.reactor_comp.inputs[4] == m.reactor_ch3oh.inputs[4])
m.con26 = pyo.Constraint(expr=m.reactor_comp.inputs[5] == m.reactor_ch3oh.inputs[5])
m.con27 = pyo.Constraint(expr=m.reactor_comp.inputs[6] == m.reactor_ch3oh.inputs[6])
m.con28 = pyo.Constraint(expr=m.reactor_comp.inputs[7] == m.reactor_ch3oh.inputs[7])

m.con29 = pyo.Constraint(expr=m.reactor_h2.inputs[0] == m.reactor_ch3oh.inputs[0])
m.con30 = pyo.Constraint(expr=m.reactor_h2.inputs[1] == m.reactor_ch3oh.inputs[1])
m.con31 = pyo.Constraint(expr=m.reactor_h2.inputs[2] == m.reactor_ch3oh.inputs[2])
m.con32 = pyo.Constraint(expr=m.reactor_h2.inputs[3] == m.reactor_ch3oh.inputs[3])
m.con33 = pyo.Constraint(expr=m.reactor_h2.inputs[4] == m.reactor_ch3oh.inputs[4])
m.con34 = pyo.Constraint(expr=m.reactor_h2.inputs[5] == m.reactor_ch3oh.inputs[5])
m.con35 = pyo.Constraint(expr=m.reactor_h2.inputs[6] == m.reactor_ch3oh.inputs[6])
m.con36 = pyo.Constraint(expr=m.reactor_h2.inputs[7] == m.reactor_ch3oh.inputs[7])

m.con37 = pyo.Constraint(expr=m.reactor_h2.inputs[0] == m.reactor_T.inputs[0])
m.con38 = pyo.Constraint(expr=m.reactor_h2.inputs[1] == m.reactor_T.inputs[1])
m.con39 = pyo.Constraint(expr=m.reactor_h2.inputs[2] == m.reactor_T.inputs[2])
m.con40 = pyo.Constraint(expr=m.reactor_h2.inputs[3] == m.reactor_T.inputs[3])
m.con41 = pyo.Constraint(expr=m.reactor_h2.inputs[4] == m.reactor_T.inputs[4])
m.con42 = pyo.Constraint(expr=m.reactor_h2.inputs[5] == m.reactor_T.inputs[5])
m.con43 = pyo.Constraint(expr=m.reactor_h2.inputs[6] == m.reactor_T.inputs[6])
m.con44 = pyo.Constraint(expr=m.reactor_h2.inputs[7] == m.reactor_T.inputs[7])

m.con45 = pyo.Constraint(expr=m.reactor_P.inputs[0] == m.reactor_T.inputs[0])
m.con46 = pyo.Constraint(expr=m.reactor_P.inputs[1] == m.reactor_T.inputs[1])
m.con47 = pyo.Constraint(expr=m.reactor_P.inputs[2] == m.reactor_T.inputs[2])
m.con48 = pyo.Constraint(expr=m.reactor_P.inputs[3] == m.reactor_T.inputs[3])
m.con49 = pyo.Constraint(expr=m.reactor_P.inputs[4] == m.reactor_T.inputs[4])
m.con50 = pyo.Constraint(expr=m.reactor_P.inputs[5] == m.reactor_T.inputs[5])
m.con51 = pyo.Constraint(expr=m.reactor_P.inputs[6] == m.reactor_T.inputs[6])
m.con52 = pyo.Constraint(expr=m.reactor_P.inputs[7] == m.reactor_T.inputs[7])

m.con173 = pyo.Constraint(expr=m.reactor_comp.inputs[0]==m.mixer_reactor.inputs[0]+m.mixer_reactor.inputs[6])
m.con174 = pyo.Constraint(expr=m.reactor_comp.inputs[1]==m.mixer_reactor.inputs[1]+m.mixer_reactor.inputs[7])
m.con175 = pyo.Constraint(expr=m.reactor_comp.inputs[2]==m.mixer_reactor.inputs[2]+m.mixer_reactor.inputs[8])
m.con176 = pyo.Constraint(expr=m.reactor_comp.inputs[3]==m.mixer_reactor.inputs[3]+m.mixer_reactor.inputs[9])
m.con177 = pyo.Constraint(expr=m.reactor_comp.inputs[4]==m.mixer_reactor.inputs[4]+m.mixer_reactor.inputs[10])
m.con178 = pyo.Constraint(expr=m.reactor_comp.inputs[6]==m.mixer_reactor.inputs[12])
m.con179 = pyo.Constraint(expr=m.reactor_comp.inputs[7]>=42)
m.con1791 = pyo.Constraint(expr=m.reactor_comp.inputs[7]<=60) 

#%%
# cooler equate
m.con53 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[0] == m.cooler_vap_h2.inputs[0])
m.con54 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[1] == m.cooler_vap_h2.inputs[1])
m.con55 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[2] == m.cooler_vap_h2.inputs[2])
m.con56 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[3] == m.cooler_vap_h2.inputs[3])
m.con57 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[4] == m.cooler_vap_h2.inputs[4])
m.con58 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[5] == m.cooler_vap_h2.inputs[5])
m.con59 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[6] == m.cooler_vap_h2.inputs[6])
m.con60 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[7] == m.cooler_vap_h2.inputs[7])

m.con69 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[0] == m.cooler_liq_ch3oh_h2o.inputs[0])
m.con70 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[1] == m.cooler_liq_ch3oh_h2o.inputs[1])
m.con71 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[2] == m.cooler_liq_ch3oh_h2o.inputs[2])
m.con72 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[3] == m.cooler_liq_ch3oh_h2o.inputs[3])
m.con73 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[4] == m.cooler_liq_ch3oh_h2o.inputs[4])
m.con74 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[5] == m.cooler_liq_ch3oh_h2o.inputs[5])
m.con75 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[6] == m.cooler_liq_ch3oh_h2o.inputs[6])
m.con76 = pyo.Constraint(expr=m.cooler_vap_h2.inputs[7] == m.cooler_liq_ch3oh_h2o.inputs[7])


m.con109 = pyo.Constraint(expr=m.cooler_power.inputs[0] == m.cooler_liq_ch3oh_h2o.inputs[0])
m.con110 = pyo.Constraint(expr=m.cooler_power.inputs[1] == m.cooler_liq_ch3oh_h2o.inputs[1])
m.con111 = pyo.Constraint(expr=m.cooler_power.inputs[2] == m.cooler_liq_ch3oh_h2o.inputs[2])
m.con112 = pyo.Constraint(expr=m.cooler_power.inputs[3] == m.cooler_liq_ch3oh_h2o.inputs[3])
m.con113 = pyo.Constraint(expr=m.cooler_power.inputs[4] == m.cooler_liq_ch3oh_h2o.inputs[4])
m.con114 = pyo.Constraint(expr=m.cooler_power.inputs[5] == m.cooler_liq_ch3oh_h2o.inputs[5])
m.con115 = pyo.Constraint(expr=m.cooler_power.inputs[6] == m.cooler_liq_ch3oh_h2o.inputs[6])
m.con116 = pyo.Constraint(expr=m.cooler_power.inputs[7] == m.cooler_liq_ch3oh_h2o.inputs[7])

m.con117 = pyo.Constraint(expr=m.cooler_power.inputs[0] == m.cooler_VD.inputs[0])
m.con118 = pyo.Constraint(expr=m.cooler_power.inputs[1] == m.cooler_VD.inputs[1])
m.con119 = pyo.Constraint(expr=m.cooler_power.inputs[2] == m.cooler_VD.inputs[2])
m.con120 = pyo.Constraint(expr=m.cooler_power.inputs[3] == m.cooler_VD.inputs[3])
m.con121 = pyo.Constraint(expr=m.cooler_power.inputs[4] == m.cooler_VD.inputs[4])
m.con122 = pyo.Constraint(expr=m.cooler_power.inputs[5] == m.cooler_VD.inputs[5])
m.con123 = pyo.Constraint(expr=m.cooler_power.inputs[6] == m.cooler_VD.inputs[6])
m.con124 = pyo.Constraint(expr=m.cooler_power.inputs[7] == m.cooler_VD.inputs[7])

m.con125 = pyo.Constraint(expr=m.cooler_LD.inputs[0] == m.cooler_VD.inputs[0])
m.con126 = pyo.Constraint(expr=m.cooler_LD.inputs[1] == m.cooler_VD.inputs[1])
m.con127 = pyo.Constraint(expr=m.cooler_LD.inputs[2] == m.cooler_VD.inputs[2])
m.con128 = pyo.Constraint(expr=m.cooler_LD.inputs[3] == m.cooler_VD.inputs[3])
m.con129 = pyo.Constraint(expr=m.cooler_LD.inputs[4] == m.cooler_VD.inputs[4])
m.con130 = pyo.Constraint(expr=m.cooler_LD.inputs[5] == m.cooler_VD.inputs[5])
m.con131 = pyo.Constraint(expr=m.cooler_LD.inputs[6] == m.cooler_VD.inputs[6])
m.con132 = pyo.Constraint(expr=m.cooler_LD.inputs[7] == m.cooler_VD.inputs[7])

m.con180 = pyo.Constraint(expr=m.cooler_LD.inputs[0]==m.reactor_comp.outputs[0]*3600)
m.con181 = pyo.Constraint(expr=m.cooler_LD.inputs[1]==m.reactor_h2.outputs[0]*3600)
m.con182 = pyo.Constraint(expr=m.cooler_LD.inputs[2]==m.reactor_ch3oh.outputs[0]*3600)
m.con183 = pyo.Constraint(expr=m.cooler_LD.inputs[3]==m.reactor_comp.outputs[1]*3600)
m.con184 = pyo.Constraint(expr=m.cooler_LD.inputs[4]==m.reactor_comp.outputs[2]*3600)
m.con185 = pyo.Constraint(expr=m.cooler_LD.inputs[5]==m.reactor_T.outputs[0])
m.con186 = pyo.Constraint(expr=m.cooler_LD.inputs[6]==m.reactor_P.outputs[0]) 

m.con187 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[5]-m.cooler_vap_co2_co.inputs[7]>=20) 
m.con188 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[5]-m.cooler_vap_co2_co.inputs[7]<=50)

#%%
# compressor recycle equate
m.con133 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[0] == m.compressor_Mix_E.inputs[0])
m.con134 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[1] == m.compressor_Mix_E.inputs[1])
m.con135 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[2] == m.compressor_Mix_E.inputs[2])
m.con136 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[3] == m.compressor_Mix_E.inputs[3])
m.con137 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[4] == m.compressor_Mix_E.inputs[4])
m.con138 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[5] == m.compressor_Mix_E.inputs[5])
m.con139 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[6] == m.compressor_Mix_E.inputs[6])
m.con140 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[7] == m.compressor_Mix_E.inputs[7])

#input to recycle compressor
m.con189 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[0] == m.cooler_vap_co2_co.outputs[0]*3600*0.99)
m.con190 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[1] == m.cooler_vap_h2.outputs[0]*3600*0.99)
m.con191 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[2] == (m.cooler_vap_co2_co.inputs[2]-m.cooler_liq_ch3oh_h2o.outputs[0]*3600)*0.99)
m.con192 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[3] == m.cooler_vap_co2_co.outputs[1]*3600*0.99)
m.con193 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[4] == (m.cooler_vap_co2_co.inputs[4]-m.cooler_liq_ch3oh_h2o.outputs[1]*3600)*0.99)
m.con194 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[5] == m.cooler_vap_co2_co.inputs[5]-m.cooler_vap_co2_co.inputs[7])
m.con195 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[6] == m.cooler_vap_co2_co.inputs[6]-100) 
m.con196 = pyo.Constraint(expr=m.compressor_Mix_T.inputs[7]*m.compressor_Mix_T.inputs[6] == m.mixer_feed.inputs[4])

#%%
#adding post recycle constraints here
#defining valve
m.con208=pyo.Constraint(expr=m.valve_vap_co.inputs[0] == m.valve_liq_ch3oh_h2o.inputs[0])
m.con209=pyo.Constraint(expr=m.valve_vap_co.inputs[1] == m.valve_liq_ch3oh_h2o.inputs[1])
m.con210=pyo.Constraint(expr=m.valve_vap_co.inputs[2] == m.valve_liq_ch3oh_h2o.inputs[2])
m.con211=pyo.Constraint(expr=m.valve_vap_co.inputs[3] == m.valve_liq_ch3oh_h2o.inputs[3])
m.con212=pyo.Constraint(expr=m.valve_vap_co.inputs[4] == m.valve_liq_ch3oh_h2o.inputs[4])
m.con213=pyo.Constraint(expr=m.valve_vap_co.inputs[5] == m.valve_liq_ch3oh_h2o.inputs[5])
m.con214=pyo.Constraint(expr=m.valve_vap_co.inputs[6] == m.valve_liq_ch3oh_h2o.inputs[6])
m.con215=pyo.Constraint(expr=m.valve_vap_co.inputs[7] == m.valve_liq_ch3oh_h2o.inputs[7])

m.con232=pyo.Constraint(expr=m.valve_liq_co2.inputs[0] == m.valve_liq_ch3oh_h2o.inputs[0])
m.con233=pyo.Constraint(expr=m.valve_liq_co2.inputs[1] == m.valve_liq_ch3oh_h2o.inputs[1])
m.con234=pyo.Constraint(expr=m.valve_liq_co2.inputs[2] == m.valve_liq_ch3oh_h2o.inputs[2])
m.con235=pyo.Constraint(expr=m.valve_liq_co2.inputs[3] == m.valve_liq_ch3oh_h2o.inputs[3])
m.con236=pyo.Constraint(expr=m.valve_liq_co2.inputs[4] == m.valve_liq_ch3oh_h2o.inputs[4])
m.con237=pyo.Constraint(expr=m.valve_liq_co2.inputs[5] == m.valve_liq_ch3oh_h2o.inputs[5])
m.con238=pyo.Constraint(expr=m.valve_liq_co2.inputs[6] == m.valve_liq_ch3oh_h2o.inputs[6])
m.con239=pyo.Constraint(expr=m.valve_liq_co2.inputs[7] == m.valve_liq_ch3oh_h2o.inputs[7])

m.con248=pyo.Constraint(expr=m.valve_liq_co2.inputs[0] == m.valve_T.inputs[0])
m.con249=pyo.Constraint(expr=m.valve_liq_co2.inputs[1] == m.valve_T.inputs[1])
m.con250=pyo.Constraint(expr=m.valve_liq_co2.inputs[2] == m.valve_T.inputs[2])
m.con251=pyo.Constraint(expr=m.valve_liq_co2.inputs[3] == m.valve_T.inputs[3])
m.con252=pyo.Constraint(expr=m.valve_liq_co2.inputs[4] == m.valve_T.inputs[4])
m.con253=pyo.Constraint(expr=m.valve_liq_co2.inputs[5] == m.valve_T.inputs[5])
m.con254=pyo.Constraint(expr=m.valve_liq_co2.inputs[6] == m.valve_T.inputs[6])
m.con255=pyo.Constraint(expr=m.valve_liq_co2.inputs[7] == m.valve_T.inputs[7])

m.con256=pyo.Constraint(expr=m.valve_VD.inputs[0] == m.valve_T.inputs[0])
m.con257=pyo.Constraint(expr=m.valve_VD.inputs[1] == m.valve_T.inputs[1])
m.con258=pyo.Constraint(expr=m.valve_VD.inputs[2] == m.valve_T.inputs[2])
m.con259=pyo.Constraint(expr=m.valve_VD.inputs[3] == m.valve_T.inputs[3])
m.con260=pyo.Constraint(expr=m.valve_VD.inputs[4] == m.valve_T.inputs[4])
m.con261=pyo.Constraint(expr=m.valve_VD.inputs[5] == m.valve_T.inputs[5])
m.con262=pyo.Constraint(expr=m.valve_VD.inputs[6] == m.valve_T.inputs[6])
m.con263=pyo.Constraint(expr=m.valve_VD.inputs[7] == m.valve_T.inputs[7])

m.con264=pyo.Constraint(expr=m.valve_LD.inputs[0] == m.valve_VD.inputs[0])
m.con265=pyo.Constraint(expr=m.valve_LD.inputs[1] == m.valve_VD.inputs[1])
m.con266=pyo.Constraint(expr=m.valve_LD.inputs[2] == m.valve_VD.inputs[2])
m.con267=pyo.Constraint(expr=m.valve_LD.inputs[3] == m.valve_VD.inputs[3])
m.con268=pyo.Constraint(expr=m.valve_LD.inputs[4] == m.valve_VD.inputs[4])
m.con269=pyo.Constraint(expr=m.valve_LD.inputs[5] == m.valve_VD.inputs[5])
m.con270=pyo.Constraint(expr=m.valve_LD.inputs[6] == m.valve_VD.inputs[6])
m.con271=pyo.Constraint(expr=m.valve_LD.inputs[7] == m.valve_VD.inputs[7])

#inputs to valve 
m.con272=pyo.Constraint(expr=m.valve_VD.inputs[0] == m.cooler_vap_co2_co.inputs[0]-m.cooler_vap_co2_co.outputs[0]*3600)
m.con273=pyo.Constraint(expr=m.valve_VD.inputs[1] == m.cooler_vap_co2_co.inputs[1]-m.cooler_vap_h2.outputs[0]*3600)
m.con274=pyo.Constraint(expr=m.valve_VD.inputs[2] == m.cooler_liq_ch3oh_h2o.outputs[0]*3600)
m.con275=pyo.Constraint(expr=m.valve_VD.inputs[3] == m.cooler_vap_co2_co.inputs[3]-m.cooler_vap_co2_co.outputs[1]*3600)
m.con276=pyo.Constraint(expr=m.valve_VD.inputs[4] == m.cooler_liq_ch3oh_h2o.outputs[1]*3600)
m.con277=pyo.Constraint(expr=m.valve_VD.inputs[5] == m.cooler_vap_co2_co.inputs[5]-m.cooler_vap_co2_co.inputs[7])
m.con278=pyo.Constraint(expr=m.valve_VD.inputs[6] == m.cooler_vap_co2_co.inputs[6]-100)
m.con300=pyo.Constraint(expr=(m.valve_VD.inputs[6]-m.valve_VD.inputs[7])>=101) 
m.con3001=pyo.Constraint(expr=(m.valve_VD.inputs[6]-m.valve_VD.inputs[7])<=200)
m.con3002=pyo.Constraint(expr=(m.valve_LD.inputs[2]-m.valve_liq_ch3oh_h2o.outputs[0]*3600)>=0)

#%%
#input to heater
m.con301=pyo.Constraint(expr=m.heater.inputs[0] == m.valve_liq_co2.outputs[0]*3600)
m.con302=pyo.Constraint(expr=m.heater.inputs[1] == 0)
m.con303=pyo.Constraint(expr=m.heater.inputs[2] == m.valve_liq_ch3oh_h2o.outputs[0]*3600)
m.con304=pyo.Constraint(expr=m.heater.inputs[3] == m.valve_VD.inputs[3]-m.valve_vap_co.outputs[0]*3600)
m.con305=pyo.Constraint(expr=m.heater.inputs[4] == m.valve_liq_ch3oh_h2o.outputs[1]*3600)
m.con306=pyo.Constraint(expr=m.heater.inputs[5] == m.valve_T.outputs[0])
m.con307=pyo.Constraint(expr=m.heater.inputs[6] == m.valve_VD.inputs[6]-m.valve_VD.inputs[7])
m.con308=pyo.Constraint(expr=m.heater.inputs[7] >=70)
#%%
# distillation column definition
m.con309=pyo.Constraint(expr=m.dis_vap_co2.inputs[0] == m.dis_vap_ch3oh.inputs[0])
m.con310=pyo.Constraint(expr=m.dis_vap_co2.inputs[1] == m.dis_vap_ch3oh.inputs[1])
m.con311=pyo.Constraint(expr=m.dis_vap_co2.inputs[2] == m.dis_vap_ch3oh.inputs[2])
m.con312=pyo.Constraint(expr=m.dis_vap_co2.inputs[3] == m.dis_vap_ch3oh.inputs[3])
m.con313=pyo.Constraint(expr=m.dis_vap_co2.inputs[4] == m.dis_vap_ch3oh.inputs[4])
m.con314=pyo.Constraint(expr=m.dis_vap_co2.inputs[5] == m.dis_vap_ch3oh.inputs[5])
m.con315=pyo.Constraint(expr=m.dis_vap_co2.inputs[6] == m.dis_vap_ch3oh.inputs[6])

m.con317=pyo.Constraint(expr=m.dis_vap_co.inputs[0] == m.dis_vap_ch3oh.inputs[0])
m.con318=pyo.Constraint(expr=m.dis_vap_co.inputs[1] == m.dis_vap_ch3oh.inputs[1])
m.con319=pyo.Constraint(expr=m.dis_vap_co.inputs[2] == m.dis_vap_ch3oh.inputs[2])
m.con320=pyo.Constraint(expr=m.dis_vap_co.inputs[3] == m.dis_vap_ch3oh.inputs[3])
m.con321=pyo.Constraint(expr=m.dis_vap_co.inputs[4] == m.dis_vap_ch3oh.inputs[4])
m.con322=pyo.Constraint(expr=m.dis_vap_co.inputs[5] == m.dis_vap_ch3oh.inputs[5])
m.con323=pyo.Constraint(expr=m.dis_vap_co.inputs[6] == m.dis_vap_ch3oh.inputs[6])

m.con325=pyo.Constraint(expr=m.dis_vap_co.inputs[0] == m.dis_vap_h2o.inputs[0])
m.con326=pyo.Constraint(expr=m.dis_vap_co.inputs[1] == m.dis_vap_h2o.inputs[1])
m.con327=pyo.Constraint(expr=m.dis_vap_co.inputs[2] == m.dis_vap_h2o.inputs[2])
m.con328=pyo.Constraint(expr=m.dis_vap_co.inputs[3] == m.dis_vap_h2o.inputs[3])
m.con329=pyo.Constraint(expr=m.dis_vap_co.inputs[4] == m.dis_vap_h2o.inputs[4])
m.con330=pyo.Constraint(expr=m.dis_vap_co.inputs[5] == m.dis_vap_h2o.inputs[5])
m.con331=pyo.Constraint(expr=m.dis_vap_co.inputs[6] == m.dis_vap_h2o.inputs[6])

m.con333=pyo.Constraint(expr=m.dis_con_T.inputs[0] == m.dis_vap_h2o.inputs[0])
m.con334=pyo.Constraint(expr=m.dis_con_T.inputs[1] == m.dis_vap_h2o.inputs[1])
m.con335=pyo.Constraint(expr=m.dis_con_T.inputs[2] == m.dis_vap_h2o.inputs[2])
m.con336=pyo.Constraint(expr=m.dis_con_T.inputs[3] == m.dis_vap_h2o.inputs[3])
m.con337=pyo.Constraint(expr=m.dis_con_T.inputs[4] == m.dis_vap_h2o.inputs[4])
m.con338=pyo.Constraint(expr=m.dis_con_T.inputs[5] == m.dis_vap_h2o.inputs[5])
m.con339=pyo.Constraint(expr=m.dis_con_T.inputs[6] == m.dis_vap_h2o.inputs[6])

m.con341=pyo.Constraint(expr=m.dis_con_T.inputs[0] == m.dis_reb_T.inputs[0])
m.con342=pyo.Constraint(expr=m.dis_con_T.inputs[1] == m.dis_reb_T.inputs[1])
m.con343=pyo.Constraint(expr=m.dis_con_T.inputs[2] == m.dis_reb_T.inputs[2])
m.con344=pyo.Constraint(expr=m.dis_con_T.inputs[3] == m.dis_reb_T.inputs[3])
m.con345=pyo.Constraint(expr=m.dis_con_T.inputs[4] == m.dis_reb_T.inputs[4])
m.con346=pyo.Constraint(expr=m.dis_con_T.inputs[5] == m.dis_reb_T.inputs[5])
m.con347=pyo.Constraint(expr=m.dis_con_T.inputs[6] == m.dis_reb_T.inputs[6])

m.con349=pyo.Constraint(expr=m.dis_con_T.inputs[0] == m.dis_con_duty.inputs[0])
m.con350=pyo.Constraint(expr=m.dis_con_T.inputs[1] == m.dis_con_duty.inputs[1])
m.con351=pyo.Constraint(expr=m.dis_con_T.inputs[2] == m.dis_con_duty.inputs[2])
m.con352=pyo.Constraint(expr=m.dis_con_T.inputs[3] == m.dis_con_duty.inputs[3])
m.con353=pyo.Constraint(expr=m.dis_con_T.inputs[4] == m.dis_con_duty.inputs[4])
m.con354=pyo.Constraint(expr=m.dis_con_T.inputs[5] == m.dis_con_duty.inputs[5])
m.con355=pyo.Constraint(expr=m.dis_con_T.inputs[6] == m.dis_con_duty.inputs[6])

m.con357=pyo.Constraint(expr=m.dis_reb_duty.inputs[0] == m.dis_con_duty.inputs[0])
m.con358=pyo.Constraint(expr=m.dis_reb_duty.inputs[1] == m.dis_con_duty.inputs[1])
m.con359=pyo.Constraint(expr=m.dis_reb_duty.inputs[2] == m.dis_con_duty.inputs[2])
m.con360=pyo.Constraint(expr=m.dis_reb_duty.inputs[3] == m.dis_con_duty.inputs[3])
m.con361=pyo.Constraint(expr=m.dis_reb_duty.inputs[4] == m.dis_con_duty.inputs[4])
m.con362=pyo.Constraint(expr=m.dis_reb_duty.inputs[5] == m.dis_con_duty.inputs[5])
m.con363=pyo.Constraint(expr=m.dis_reb_duty.inputs[6] == m.dis_con_duty.inputs[6])

m.con365=pyo.Constraint(expr=m.dis_reb_duty.inputs[0] == m.dis_VD.inputs[0])
m.con366=pyo.Constraint(expr=m.dis_reb_duty.inputs[1] == m.dis_VD.inputs[1])
m.con367=pyo.Constraint(expr=m.dis_reb_duty.inputs[2] == m.dis_VD.inputs[2])
m.con368=pyo.Constraint(expr=m.dis_reb_duty.inputs[3] == m.dis_VD.inputs[3])
m.con369=pyo.Constraint(expr=m.dis_reb_duty.inputs[4] == m.dis_VD.inputs[4])
m.con370=pyo.Constraint(expr=m.dis_reb_duty.inputs[5] == m.dis_VD.inputs[5])
m.con371=pyo.Constraint(expr=m.dis_reb_duty.inputs[6] == m.dis_VD.inputs[6])

m.con373=pyo.Constraint(expr=m.dis_LD.inputs[0] == m.dis_VD.inputs[0])
m.con374=pyo.Constraint(expr=m.dis_LD.inputs[1] == m.dis_VD.inputs[1])
m.con375=pyo.Constraint(expr=m.dis_LD.inputs[2] == m.dis_VD.inputs[2])
m.con376=pyo.Constraint(expr=m.dis_LD.inputs[3] == m.dis_VD.inputs[3])
m.con377=pyo.Constraint(expr=m.dis_LD.inputs[4] == m.dis_VD.inputs[4])
m.con378=pyo.Constraint(expr=m.dis_LD.inputs[5] == m.dis_VD.inputs[5])
m.con379=pyo.Constraint(expr=m.dis_LD.inputs[6] == m.dis_VD.inputs[6])

m.con381=pyo.Constraint(expr=m.dis_LD.inputs[0] == m.dis_max_vap.inputs[0])
m.con382=pyo.Constraint(expr=m.dis_LD.inputs[1] == m.dis_max_vap.inputs[1])
m.con383=pyo.Constraint(expr=m.dis_LD.inputs[2] == m.dis_max_vap.inputs[2])
m.con384=pyo.Constraint(expr=m.dis_LD.inputs[3] == m.dis_max_vap.inputs[3])
m.con385=pyo.Constraint(expr=m.dis_LD.inputs[4] == m.dis_max_vap.inputs[4])
m.con386=pyo.Constraint(expr=m.dis_LD.inputs[5] == m.dis_max_vap.inputs[5])
m.con387=pyo.Constraint(expr=m.dis_LD.inputs[6] == m.dis_max_vap.inputs[6])

m.con389=pyo.Constraint(expr=m.dis_no_stage.inputs[0] == m.dis_max_vap.inputs[0])
m.con390=pyo.Constraint(expr=m.dis_no_stage.inputs[1] == m.dis_max_vap.inputs[1])
m.con391=pyo.Constraint(expr=m.dis_no_stage.inputs[2] == m.dis_max_vap.inputs[2])
m.con392=pyo.Constraint(expr=m.dis_no_stage.inputs[3] == m.dis_max_vap.inputs[3])
m.con393=pyo.Constraint(expr=m.dis_no_stage.inputs[4] == m.dis_max_vap.inputs[4])
m.con394=pyo.Constraint(expr=m.dis_no_stage.inputs[5] == m.dis_max_vap.inputs[5])
m.con395=pyo.Constraint(expr=m.dis_no_stage.inputs[6] == m.dis_max_vap.inputs[6])

m.con397=pyo.Constraint(expr=m.dis_vap_co2.inputs[0] == m.heater.inputs[0])
m.con398=pyo.Constraint(expr=m.dis_vap_co2.inputs[1] == m.heater.inputs[1])
m.con399=pyo.Constraint(expr=m.dis_vap_co2.inputs[2] == m.heater.inputs[2])
m.con400=pyo.Constraint(expr=m.dis_vap_co2.inputs[3] == m.heater.inputs[3])
m.con401=pyo.Constraint(expr=m.dis_vap_co2.inputs[4] == m.heater.inputs[4])
m.con402=pyo.Constraint(expr=m.dis_vap_co2.inputs[5] == m.heater.inputs[7])
m.con403=pyo.Constraint(expr=m.dis_vap_co2.inputs[6] == m.heater.inputs[6]-10)
m.con404=pyo.Constraint(expr=m.dis_vap_ch3oh.outputs[0] <= m.heater.inputs[2])
m.con405=pyo.Constraint(expr=m.dis_vap_co2.outputs[0]*3600 <= m.heater.inputs[0])
m.con406=pyo.Constraint(expr=m.dis_vap_co.outputs[0]*3600 <= m.heater.inputs[3])

m.con407=pyo.Constraint(expr=m.dis_no_stage.inputs[0] == m.dis_feed_stage.inputs[0])
m.con408=pyo.Constraint(expr=m.dis_no_stage.inputs[1] == m.dis_feed_stage.inputs[1])
m.con409=pyo.Constraint(expr=m.dis_no_stage.inputs[2] == m.dis_feed_stage.inputs[2])
m.con410=pyo.Constraint(expr=m.dis_no_stage.inputs[3] == m.dis_feed_stage.inputs[3])
m.con411=pyo.Constraint(expr=m.dis_no_stage.inputs[4] == m.dis_feed_stage.inputs[4])
m.con412=pyo.Constraint(expr=m.dis_no_stage.inputs[5] == m.dis_feed_stage.inputs[5])
m.con413=pyo.Constraint(expr=m.dis_no_stage.inputs[6] == m.dis_feed_stage.inputs[6])

#%%%

m.con414 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[0] == m.cooler_liq_co2_2.inputs[0])
m.con415 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[1] == m.cooler_liq_co2_2.inputs[1])
m.con416 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[2] == m.cooler_liq_co2_2.inputs[2])
m.con417 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[3] == m.cooler_liq_co2_2.inputs[3])
m.con418 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[4] == m.cooler_liq_co2_2.inputs[4])
m.con419 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[5] == m.cooler_liq_co2_2.inputs[5])
m.con420 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[6] == m.cooler_liq_co2_2.inputs[6])
m.con421 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[7] == m.cooler_liq_co2_2.inputs[7])

m.con422 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[0] == m.cooler_liq_co_2.inputs[0])
m.con423 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[1] == m.cooler_liq_co_2.inputs[1])
m.con424 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[2] == m.cooler_liq_co_2.inputs[2])
m.con425 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[3] == m.cooler_liq_co_2.inputs[3])
m.con426 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[4] == m.cooler_liq_co_2.inputs[4])
m.con427 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[5] == m.cooler_liq_co_2.inputs[5])
m.con428 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[6] == m.cooler_liq_co_2.inputs[6])
m.con429 = pyo.Constraint(expr=m.cooler_liq_co2_2.inputs[7] == m.cooler_liq_co_2.inputs[7])

m.con430 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[0] == m.cooler_liq_co_2.inputs[0])
m.con431 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[1] == m.cooler_liq_co_2.inputs[1])
m.con432 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[2] == m.cooler_liq_co_2.inputs[2])
m.con433 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[3] == m.cooler_liq_co_2.inputs[3])
m.con434 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[4] == m.cooler_liq_co_2.inputs[4])
m.con435 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[5] == m.cooler_liq_co_2.inputs[5])
m.con436 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[6] == m.cooler_liq_co_2.inputs[6])
m.con437 = pyo.Constraint(expr=m.cooler_liq_h2o_2.inputs[7] == m.cooler_liq_co_2.inputs[7])

m.con438 = pyo.Constraint(expr=m.cooler_power_2.inputs[0] == m.cooler_liq_h2o_2.inputs[0])
m.con439 = pyo.Constraint(expr=m.cooler_power_2.inputs[1] == m.cooler_liq_h2o_2.inputs[1])
m.con440 = pyo.Constraint(expr=m.cooler_power_2.inputs[2] == m.cooler_liq_h2o_2.inputs[2])
m.con441 = pyo.Constraint(expr=m.cooler_power_2.inputs[3] == m.cooler_liq_h2o_2.inputs[3])
m.con442 = pyo.Constraint(expr=m.cooler_power_2.inputs[4] == m.cooler_liq_h2o_2.inputs[4])
m.con443 = pyo.Constraint(expr=m.cooler_power_2.inputs[5] == m.cooler_liq_h2o_2.inputs[5])
m.con444 = pyo.Constraint(expr=m.cooler_power_2.inputs[6] == m.cooler_liq_h2o_2.inputs[6])
m.con445 = pyo.Constraint(expr=m.cooler_power_2.inputs[7] == m.cooler_liq_h2o_2.inputs[7])

m.con446 = pyo.Constraint(expr=m.cooler_power_2.inputs[0] == m.cooler_VD_2.inputs[0])
m.con447 = pyo.Constraint(expr=m.cooler_power_2.inputs[1] == m.cooler_VD_2.inputs[1])
m.con448 = pyo.Constraint(expr=m.cooler_power_2.inputs[2] == m.cooler_VD_2.inputs[2])
m.con449 = pyo.Constraint(expr=m.cooler_power_2.inputs[3] == m.cooler_VD_2.inputs[3])
m.con450 = pyo.Constraint(expr=m.cooler_power_2.inputs[4] == m.cooler_VD_2.inputs[4])
m.con451 = pyo.Constraint(expr=m.cooler_power_2.inputs[5] == m.cooler_VD_2.inputs[5])
m.con452 = pyo.Constraint(expr=m.cooler_power_2.inputs[6] == m.cooler_VD_2.inputs[6])
m.con453 = pyo.Constraint(expr=m.cooler_power_2.inputs[7] == m.cooler_VD_2.inputs[7])

m.con454 = pyo.Constraint(expr=m.cooler_LD_2.inputs[0] == m.cooler_VD_2.inputs[0])
m.con455 = pyo.Constraint(expr=m.cooler_LD_2.inputs[1] == m.cooler_VD_2.inputs[1])
m.con456 = pyo.Constraint(expr=m.cooler_LD_2.inputs[2] == m.cooler_VD_2.inputs[2])
m.con457 = pyo.Constraint(expr=m.cooler_LD_2.inputs[3] == m.cooler_VD_2.inputs[3])
m.con458 = pyo.Constraint(expr=m.cooler_LD_2.inputs[4] == m.cooler_VD_2.inputs[4])
m.con459 = pyo.Constraint(expr=m.cooler_LD_2.inputs[5] == m.cooler_VD_2.inputs[5])
m.con460 = pyo.Constraint(expr=m.cooler_LD_2.inputs[6] == m.cooler_VD_2.inputs[6])
m.con461 = pyo.Constraint(expr=m.cooler_LD_2.inputs[7] == m.cooler_VD_2.inputs[7])

m.con462 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[0] == m.dis_vap_co2.outputs[0]*3600)
m.con463 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[1] == 0)
m.con464 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[2] == m.dis_vap_ch3oh.outputs[0])
m.con465 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[3] == m.dis_vap_co.outputs[0]*3600)
m.con466 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[4] == m.dis_vap_h2o.outputs[0]*3600)
m.con467 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[5] == m.dis_con_T.outputs[0])
m.con468 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.inputs[6] == m.dis_vap_co2.inputs[6]-0.7*(m.dis_feed_stage.outputs[0]-1))

m.con470 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.outputs[0] >= 0)
m.con471 = pyo.Constraint(expr=m.cooler_liq_co2_2.outputs[0] >= 0)
m.con472 = pyo.Constraint(expr=m.cooler_liq_co_2.outputs[0] >= 0)
m.con473 = pyo.Constraint(expr=m.cooler_liq_h2o_2.outputs[0] >= 0)
m.con474 = pyo.Constraint(expr=m.cooler_LD_2.outputs[0] >= 0)
m.con475 = pyo.Constraint(expr=m.cooler_VD_2.outputs[0] >= 0)
m.con476 = pyo.Constraint(expr=m.cooler_power_2.outputs[0] >= 0)
m.con477 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.outputs[0]-0.96*(m.cooler_liq_ch3oh_2.outputs[0]+m.cooler_liq_co2_2.outputs[0]+m.cooler_liq_co_2.outputs[0]+m.cooler_liq_h2o_2.outputs[0])>=0)
m.con478 = pyo.Constraint(expr=m.cooler_liq_ch3oh_2.outputs[0]-0.997*m.dis_vap_ch3oh.outputs[0] <=0) 

#%%

m.con500=pyo.Constraint(expr=m.compc1==(10.406*m.compressor_E1.outputs[0]+14853.1)*800.8/(478.6*1000))
m.con501=pyo.Constraint(expr=m.compc2==(10.406*m.compressor_E2.outputs[0]+14853.1)*800.8/(478.6*1000))
m.con502=pyo.Constraint(expr=m.compc3==(10.406*m.compressor_E3.outputs[0]+14853.1)*800.8/(478.6*1000))
m.con503=pyo.Constraint(expr=m.compc4==(10.406*m.compressor_E4.outputs[0]+14853.1)*800.8/(478.6*1000))
m.con504=pyo.Constraint(expr=m.compc5==(6.4992*m.compressor_H2_E.outputs[0]+29125.58)*800.8/(478.6*1000))
m.con505=pyo.Constraint(expr=m.compc6==(5.4664*m.compressor_Mix_E.outputs[0]+37426.545)*800.8/(478.6*1000))
m.con506=pyo.Constraint(expr=m.reactorc==(0.025*m.reactor_comp.inputs[7]+30272.58)*800.8/(478.6*1000))
m.con507=pyo.Constraint(expr=m.vesselc4==(195*m.dis_no_stage.outputs[0]+26325)*800.8/(478.6*1000))

#%%
"""
m.con507=pyo.Constraint(expr=m.FV1==m.cooler_vap_co2_co.outputs[0]*44+m.cooler_vap_h2.outputs[0]*2+(m.cooler_liq_ch3oh_h2o.inputs[2]/3600-m.cooler_liq_ch3oh_h2o.outputs[0])*32+m.cooler_vap_co2_co.outputs[1]*28+(m.cooler_liq_ch3oh_h2o.inputs[4]/3600-m.cooler_liq_ch3oh_h2o.outputs[1])*18)
m.con508=pyo.Constraint(expr=m.v_max1==0.04*((m.cooler_LD_out-m.cooler_VD.outputs[0])/m.cooler_VD_out)**0.5)
m.con509=pyo.Constraint(expr=m.V_flow1*m.cooler_VD.outputs[0]==m.FV1)
m.con510=pyo.Constraint(expr=m.area1==m.V_flow1/m.v_max1)
m.con511=pyo.Constraint(expr=m.D1==(4*m.area1/3.141592653589793)**0.5)
m.con512=pyo.Constraint(expr=m.t1==((m.cooler_vap_co2_co.inputs[6]*m.D1/1000)/(2*(250*0.9+0.4*m.cooler_vap_co2_co.inputs[6]/1000)))/(1-12.5/100))
m.con513=pyo.Constraint(expr=m.W1==3.141592653589793*m.D1*3*m.D1*m.t1*7850)
m.con514=pyo.Constraint(expr=m.vesselc1==(-10000+600*(m.W1)**0.6)*(800.8/478.6))

m.con515=pyo.Constraint(expr=m.FV2==(m.valve_liq_co2.inputs[0]/3600-m.valve_liq_co2.outputs[0])*44+(m.valve_liq_co2.inputs[1]/3600)*2+(m.valve_liq_co2.inputs[2]/3600-m.valve_liq_ch3oh_h2o.outputs[0])*32+m.valve_vap_co.outputs[0]*28+(m.valve_liq_co2.inputs[4]/3600-m.valve_liq_ch3oh_h2o.outputs[1])*18)
m.con516=pyo.Constraint(expr=m.v_max2==0.04*((m.valve_LD_out-m.valve_VD.outputs[0])/m.valve_VD_out)**0.5)
m.con517=pyo.Constraint(expr=m.V_flow2*m.valve_VD.outputs[0]==m.FV2)
m.con518=pyo.Constraint(expr=m.area2==m.V_flow2/m.v_max2)
m.con519=pyo.Constraint(expr=m.D2==(4*m.area2/3.141592653589793)**0.5)
m.con520=pyo.Constraint(expr=m.t2==((m.heater.inputs[6]*m.D2/1000)/(2*(250*0.9+0.4*m.heater.inputs[6]/1000)))/(1-12.5/100))
m.con521=pyo.Constraint(expr=m.W2==3.141592653589793*m.D2*3*m.D2*m.t2*7850)
m.con522=pyo.Constraint(expr=m.vesselc2==(-10000+600*(m.W2)**0.6)*(800.8/478.6))

m.con523=pyo.Constraint(expr=m.FV3==m.dis_max_vap.outputs[0]*32/3600)
m.con524=pyo.Constraint(expr=m.v_max3==0.04*((m.dis_LD_out-m.dis_VD.outputs[0])/m.dis_VD_out)**0.5)
m.con525=pyo.Constraint(expr=m.V_flow3*m.dis_VD.outputs[0]==m.FV3)
m.con526=pyo.Constraint(expr=m.area3==m.V_flow3/m.v_max3)
m.con527=pyo.Constraint(expr=m.D3==(4*m.area3/3.141592653589793)**0.5)
m.con528=pyo.Constraint(expr=m.t3==(((m.dis_vap_co2.inputs[6]+0.7*(m.dis_no_stage.outputs[0]-m.dis_feed_stage.outputs[0]))*m.D3/1000)/(2*(250*0.9+0.4*(m.dis_vap_co2.inputs[6]+0.7*(m.dis_no_stage.outputs[0]-m.dis_feed_stage.outputs[0]))/1000)))/(1-12.5/100))
m.con529=pyo.Constraint(expr=m.W3==3.141592653589793*m.D3*3*m.D3*m.t3*7850)
m.con530=pyo.Constraint(expr=m.vesselc3==((-10000+600*(m.W3)**0.6)+(100+120*(m.D3)**2)*m.dis_no_stage.outputs[0])*(800.8/478.6))
https://www.osti.gov/servlets/purl/797810
m.con531=pyo.Constraint(expr=m.FV4==(m.cooler_liq_co2_2.outputs[0]*44+m.cooler_liq_ch3oh_2.outputs[0]*32+m.cooler_liq_co_2.outputs[0]*28+m.cooler_liq_h2o_2.outputs[0]*18)/3600)
m.con532=pyo.Constraint(expr=m.v_max4==0.04*((m.cooler_LD_2_out-m.cooler_VD_2.outputs[0])/m.cooler_VD_2_out)**0.5)
m.con533=pyo.Constraint(expr=m.V_flow4*m.cooler_VD_2.outputs[0]==m.FV4)
m.con534=pyo.Constraint(expr=m.area4==m.V_flow4/m.v_max4)
m.con535=pyo.Constraint(expr=m.D4==(4*m.area4/3.141592653589793)**0.5)
m.con536=pyo.Constraint(expr=m.t4==(((m.cooler_liq_ch3oh_2.inputs[6])*m.D4/1000)/(2*(250*0.9+0.4*m.cooler_liq_ch3oh_2.inputs[6]/1000)))/(1-12.5/100))
m.con537=pyo.Constraint(expr=m.W4==3.141592653589793*m.D4*3*m.D4*m.t4*7850)
m.con538=pyo.Constraint(expr=m.vesselc4==(-10000+600*(m.W4)**0.6)*(800.8/478.6))

m.capex_con=pyo.Constraint(expr=m.total_capex==6/(8000)*((m.compc1+m.compc2+m.compc3+m.compc4+m.compc5+m.compc6+m.reactorc+m.vesselc1+m.vesselc2+m.vesselc3+m.vesselc4))))
"""
#%%

m.capex_con=pyo.Constraint(expr=m.total_capex==600000/(8000*3600)*(m.compc1+m.compc2+m.compc3+m.compc4+m.compc5+m.compc6+m.reactorc+m.vesselc4))
m.totalE_con=pyo.Constraint(expr=m.Total_Energy==m.compressor_E1.outputs[0]+m.compressor_E2.outputs[0]+m.compressor_E3.outputs[0]+m.compressor_E4.outputs[0]+m.compressor_H2_E.outputs[0]+m.compressor_Mix_E.outputs[0])
m.opex_con=pyo.Constraint(expr=m.total_opex == (2.0722*(m.compressor_E1.inputs[0]*44*0.045+m.compressor_H2_E.inputs[0]*2*0.473)+m.Total_Energy*0.25)/3600)
m.check14=pyo.Constraint(expr=m.cooler_liq_ch3oh_2_out == m.cooler_liq_ch3oh_2.outputs[0])                          
m.TAC_con=pyo.Constraint(expr=m.TAC==(m.total_capex/100+m.total_opex)/(m.cooler_liq_ch3oh_2_out*32/3600))


#%%
"""
m.check1=pyo.Constraint(expr=m.compressor_E1_out == m.compressor_E1.outputs[0])
m.check2=pyo.Constraint(expr=m.compressor_E2_out == m.compressor_E2.outputs[0])
m.check3=pyo.Constraint(expr=m.compressor_E3_out == m.compressor_E3.outputs[0])
m.check4=pyo.Constraint(expr=m.compressor_E4_out == m.compressor_E4.outputs[0])
m.check5=pyo.Constraint(expr=m.compressor_H2_E_out == m.compressor_H2_E.outputs[0])
m.check6=pyo.Constraint(expr=m.compressor_Mix_E_out == m.compressor_Mix_E.outputs[0])
m.check7=pyo.Constraint(expr=m.reactor_comp_in == m.reactor_comp.inputs[7])
m.check8=pyo.Constraint(expr=m.cooler_VD_out == m.cooler_VD.outputs[0])
m.check9=pyo.Constraint(expr=m.cooler_LD_out == m.cooler_LD.outputs[0])

m.check10=pyo.Constraint(expr=m.valve_VD_out == m.valve_VD.outputs[0])
m.check11=pyo.Constraint(expr=m.valve_LD_out == m.valve_LD.outputs[0])
m.check12=pyo.Constraint(expr=m.dis_VD_out == m.dis_VD.outputs[0])
m.check13=pyo.Constraint(expr=m.dis_LD_out == m.dis_LD.outputs[0])
m.check14=pyo.Constraint(expr=m.cooler_liq_ch3oh_2_out == m.cooler_liq_ch3oh_2.outputs[0])
m.check15=pyo.Constraint(expr=m.cooler_VD_2_out == m.cooler_VD_2.outputs[0])
m.check16=pyo.Constraint(expr=m.cooler_LD_2_out == m.cooler_LD_2.outputs[0])
m.check17=pyo.Constraint(expr=m.dis_vap_ch3oh_out == m.dis_vap_ch3oh.outputs[0])
"""
#%%
solver = pyo.SolverFactory('ipopt')
solver.options['tol'] = 1E-4
solver.options['max_iter'] = 100000
status = solver.solve(m, tee=True)

#%%#%%
"""m.y=pyo.Var(within=Binary)
m.y_c=pyo.Constraint(expr=m.y==1)
SolverFactory('mindtpy').solve(m,
                                   #strategy='ECP',
                                   #time_limit=3600,
                                   nlp_solver='gams',
                                   nlp_solver_args=dict(solver='baron', warmstart=True),
                                   mip_solver='gams',
                                   mip_solver_args=dict(solver='baron', warmstart=True),
                                   tee=True)"""

#%%
print('F1:', pyo.value(m.compressor_E1.inputs[0]))
print('F1:', pyo.value(m.compressor_T1.inputs[0]))
#%%
#for compressor 1
print('F1:', pyo.value(m.compressor_E1.inputs[0]))
print('T1:', pyo.value(m.compressor_E1.inputs[1]))
print('P1:', pyo.value(m.compressor_E1.inputs[2]))
print('PR:', pyo.value(m.compressor_E1.inputs[3]))
print('T2:', pyo.value(m.compressor_T1.outputs[0]))
print('E1:', pyo.value(m.compressor_E1.outputs[0]))
#%%
#for compressor 2
print('F2:', pyo.value(m.compressor_E2.inputs[0]))
print('T2_cooled:', pyo.value(m.compressor_E2.inputs[1]))
print('P2:', pyo.value(m.compressor_E2.inputs[2]))
print('PR2:', pyo.value(m.compressor_E2.inputs[3]))
print('T3:', pyo.value(m.compressor_T2.outputs[0]))
print('E2:', pyo.value(m.compressor_E2.outputs[0]))
#%%
#for compressor 3
print('F3:', pyo.value(m.compressor_E3.inputs[0]))
print('T3_cooled:', pyo.value(m.compressor_E3.inputs[1]))
print('P3:', pyo.value(m.compressor_E3.inputs[2]))
print('PR3:', pyo.value(m.compressor_E3.inputs[3]))
print('T4:', pyo.value(m.compressor_T3.outputs[0]))
print('E3:', pyo.value(m.compressor_E3.outputs[0]))
#%%
#for compressor 4
print('F4:', pyo.value(m.compressor_E4.inputs[0]))
print('T4_cooled:', pyo.value(m.compressor_E4.inputs[1]))
print('P4:', pyo.value(m.compressor_E4.inputs[2]))
print('PR4:', pyo.value(m.compressor_E4.inputs[3]))
print('T5:', pyo.value(m.compressor_T4.outputs[0]))
print('E4:', pyo.value(m.compressor_E4.outputs[0]))

#%%
#for h2 compressor
print('F1:', pyo.value(m.compressor_H2_E.inputs[0]))
print('T1:', pyo.value(m.compressor_H2_E.inputs[1]))
print('P1:', pyo.value(m.compressor_H2_E.inputs[2]))
print('PR:', pyo.value(m.compressor_H2_E.inputs[3]))
print('T2:', pyo.value(m.compressor_H2_T.outputs[0]))
print('E2:', pyo.value(m.compressor_H2_E.outputs[0]))
#%%
# for mixer 
print('F1:', pyo.value(m.mixer_feed.inputs[0]))
print('T1:', pyo.value(m.mixer_feed.inputs[1]))
print('F2:', pyo.value(m.mixer_feed.inputs[2]))
print('T2:', pyo.value(m.mixer_feed.inputs[3]))
print('P1:', pyo.value(m.mixer_feed.inputs[4]))
print('T_out:', pyo.value(m.mixer_feed.outputs[0]))

#%%
# for mixer 2
print('F1CO2:', pyo.value(m.mixer_reactor.inputs[0]))
print('F1H2:', pyo.value(m.mixer_reactor.inputs[1]))
print('F1CH3OH:', pyo.value(m.mixer_reactor.inputs[2]))
print('F1CO:', pyo.value(m.mixer_reactor.inputs[3]))
print('F1H2O:', pyo.value(m.mixer_reactor.inputs[4]))
print('F1T1:', pyo.value(m.mixer_reactor.inputs[5]))
print('F2CO2:', pyo.value(m.mixer_reactor.inputs[6]))
print('F2H2:', pyo.value(m.mixer_reactor.inputs[7]))
print('F2CH3OH:', pyo.value(m.mixer_reactor.inputs[8]))
print('F2CO:', pyo.value(m.mixer_reactor.inputs[9]))
print('F2H2O:', pyo.value(m.mixer_reactor.inputs[10]))
print('F2T1:', pyo.value(m.mixer_reactor.inputs[11]))
print('P1:', pyo.value(m.mixer_reactor.inputs[12]))
print('T_out:', pyo.value(m.mixer_reactor.outputs[0]))

#%%
# for reactor inlet
print('F1CO2:', pyo.value(m.reactor_comp.inputs[0]))
print('F1H2:', pyo.value(m.reactor_comp.inputs[1]))
print('F1CH3OH:', pyo.value(m.reactor_comp.inputs[2]))
print('F1CO:', pyo.value(m.reactor_comp.inputs[3]))
print('F1H2O:', pyo.value(m.reactor_comp.inputs[4]))
print('F1T:', pyo.value(m.reactor_comp.inputs[5]))
print('F1P:', pyo.value(m.reactor_comp.inputs[6]))
print('F1V:', pyo.value(m.reactor_comp.inputs[7]))

# for reactor outlet
print('F2CO2:', pyo.value(m.reactor_comp.outputs[0]*3600))
print('F2CO:', pyo.value(m.reactor_comp.outputs[1]*3600))
print('F2H2O:', pyo.value(m.reactor_comp.outputs[2]*3600))
print('F2CH3OH:', pyo.value(m.reactor_ch3oh.outputs[0]*3600))
print('F2H2:', pyo.value(m.reactor_h2.outputs[0]*3600))
print('F2T2:', pyo.value(m.reactor_T.outputs[0]))
print('F2P2:', pyo.value(m.reactor_P.outputs[0]))

#%%
# after cooler phase changer
print('F1CO2:', pyo.value(m.cooler_vap_co2_co.inputs[0]))
print('F1H2:', pyo.value(m.cooler_vap_co2_co.inputs[1]))
print('F1CH3OH:', pyo.value(m.cooler_vap_co2_co.inputs[2]))
print('F1CO:', pyo.value(m.cooler_vap_co2_co.inputs[3]))
print('F1H2O:', pyo.value(m.cooler_vap_co2_co.inputs[4]))
print('F1T:', pyo.value(m.cooler_vap_co2_co.inputs[5]))
print('F1P:', pyo.value(m.cooler_vap_co2_co.inputs[6]))
print('dT:', pyo.value(m.cooler_vap_co2_co.inputs[7]))

print('VF2CO2:', pyo.value(m.cooler_vap_co2_co.outputs[0]*3600))
print('VF2H2:', pyo.value(m.cooler_vap_h2.outputs[0]*3600))
print('VF2CO:', pyo.value(m.cooler_vap_co2_co.outputs[1]*3600))
print('LF2CH3OH:', pyo.value(m.cooler_liq_ch3oh_h2o.outputs[0]*3600))
print('LF2H2O:', pyo.value(m.cooler_liq_ch3oh_h2o.outputs[1]*3600))
print('Energy:', pyo.value(m.cooler_power.outputs[0]))
print('Vapour Density:', pyo.value(m.cooler_VD.outputs[0]))
print('Liquid Density:', pyo.value(m.cooler_LD.outputs[0]))

#%%
print('F1CO2:', pyo.value(m.compressor_Mix_T.inputs[0]))
print('F1H2:', pyo.value(m.compressor_Mix_T.inputs[1]))
print('F1CH3OH:', pyo.value(m.compressor_Mix_T.inputs[2]))
print('F1CO:', pyo.value(m.compressor_Mix_T.inputs[3]))
print('F1H2O:', pyo.value(m.compressor_Mix_T.inputs[4]))
print('T1:', pyo.value(m.compressor_Mix_T.inputs[5]))
print('P1:', pyo.value(m.compressor_Mix_T.inputs[6]))
print('PR:', pyo.value(m.compressor_Mix_T.inputs[7]))

print('T_out:', pyo.value(m.compressor_Mix_T.outputs[0]))
print('Power:', pyo.value(m.compressor_Mix_E.outputs[0]))

#%%
print('F1CO2:', pyo.value(m.valve_LD.inputs[0]))
print('F1H2:', pyo.value(m.valve_LD.inputs[1]))
print('F1CH3OH:', pyo.value(m.valve_LD.inputs[2]))
print('F1CO:', pyo.value(m.valve_LD.inputs[3]))
print('F1H2O:', pyo.value(m.valve_LD.inputs[4]))
print('T1:', pyo.value(m.valve_LD.inputs[5]))
print('P1:', pyo.value(m.valve_LD.inputs[6]))
print('dP:', pyo.value(m.valve_LD.inputs[7]))

print('VF2CO:', pyo.value(m.valve_vap_co.outputs[0]*3600))
print('LF2CO2:', pyo.value(m.valve_liq_co2.outputs[0]*3600))
print('LF2H2:  0')
print('LF2CH3OH:', pyo.value(m.valve_liq_ch3oh_h2o.outputs[0]*3600))
print('LF2H2O:', pyo.value(m.valve_liq_ch3oh_h2o.outputs[1]*3600))
print('T2:', pyo.value(m.valve_T.outputs[0]))
print('Vapour Density:', pyo.value(m.valve_VD.outputs[0]))
print('Liquid Density:', pyo.value(m.valve_LD.outputs[0]))
#%%
#for heater
print('F1CO2:', pyo.value(m.heater.inputs[0]))
print('F1H2:', pyo.value(m.heater.inputs[1]))
print('F1CH3OH:', pyo.value(m.heater.inputs[2]))
print('F1CO:', pyo.value(m.heater.inputs[3]))
print('F1H2O:', pyo.value(m.heater.inputs[4]))
print('T1:', pyo.value(m.heater.inputs[5]))
print('P1:', pyo.value(m.heater.inputs[6]))
print('T2:', pyo.value(m.heater.inputs[7]))

print('Heat:', pyo.value(m.heater.outputs[0]))

#%%
#for distillation column
print('F1CO2:', pyo.value(m.dis_vap_co2.inputs[0]))
print('F1H2:', pyo.value(m.dis_vap_co2.inputs[1]))
print('F1CH3OH:', pyo.value(m.dis_vap_co2.inputs[2]))
print('F1CO:', pyo.value(m.dis_vap_co2.inputs[3]))
print('F1H2O:', pyo.value(m.dis_vap_co2.inputs[4]))
print('T1:', pyo.value(m.dis_vap_co2.inputs[5]))
print('P1:', pyo.value(m.dis_vap_co2.inputs[6]))

print('VF2CO2:', pyo.value(m.dis_vap_co2.outputs[0]*3600))
print('VF2H2:', 0)
print('VF2CH3OH:', pyo.value(m.dis_vap_ch3oh.outputs[0]))
print('VF2CO:', pyo.value(m.dis_vap_co.outputs[0]*3600))
print('VF2H2O:', pyo.value(m.dis_vap_h2o.outputs[0]*3600))
print('Condensor_T:', pyo.value(m.dis_con_T.outputs[0]))
print('Reboiler_T:',pyo.value(m.dis_reb_T.outputs[0]))
print('Condensor Duty', pyo.value(m.dis_con_duty.outputs[0]))
print('Reboiler Duty:', pyo.value(m.dis_reb_duty.outputs[0]))
print('Vapour Density:', pyo.value(m.dis_VD.outputs[0]))
print('Liquid Density:', pyo.value(m.dis_LD.outputs[0]))
print('Max Vapour Flow Rate:', pyo.value(m.dis_max_vap.outputs[0]))
print('No. of Stages:', pyo.value(m.dis_no_stage.outputs[0]))
print('Feed Stage:', pyo.value(m.dis_feed_stage.outputs[0]))
#%%
#for cooler 2
print('F1CO2:', pyo.value(m.cooler_liq_ch3oh_2.inputs[0]))
print('F1H2:', pyo.value(m.cooler_liq_ch3oh_2.inputs[1]))
print('F1CH3OH:', pyo.value(m.cooler_liq_ch3oh_2.inputs[2]))
print('F1CO:', pyo.value(m.cooler_liq_ch3oh_2.inputs[3]))
print('F1H2O:', pyo.value(m.cooler_liq_ch3oh_2.inputs[4]))
print('T1:', pyo.value(m.cooler_liq_ch3oh_2.inputs[5]))
print('P1:', pyo.value(m.cooler_liq_ch3oh_2.inputs[6]))
print('T2:', pyo.value(m.cooler_liq_ch3oh_2.inputs[7]))

print('LF2CO2:', pyo.value(m.cooler_liq_co2_2.outputs[0]))
print('LF2H2:', 0)
print('LF2CH3OH:', pyo.value(m.cooler_liq_ch3oh_2.outputs[0]))
print('LF2CO:', pyo.value(m.cooler_liq_co_2.outputs[0]))
print('LF2H2O:', pyo.value(m.cooler_liq_h2o_2.outputs[0]))
print('Power:', pyo.value(m.cooler_power_2.outputs[0]))
print('Vapour Density:', pyo.value(m.cooler_VD_2.outputs[0]))
print('Liquid Density:', pyo.value(m.cooler_LD_2.outputs[0]))

#%%
print('TAC: ', pyo.value(m.TAC))
print('Opex: ', pyo.value(m.total_opex))
print('Capex: ', pyo.value(m.total_capex))
#%%
print('comp1: ', pyo.value(m.compc1))
print('comp2: ', pyo.value(m.compc2))
print('comp3: ', pyo.value(m.compc3))
print('comp4: ', pyo.value(m.compc4))
print('comp5: ', pyo.value(m.compc5))
print('comp6: ', pyo.value(m.compc6))
print('reactorc: ', pyo.value(m.reactorc))
print('vessel4: ', pyo.value(m.vesselc4))
print('Total E: ', pyo.value(m.Total_Energy))
