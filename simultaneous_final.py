import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress CUDA warnings from tensorflow

# import the necessary packages
from omlt import OmltBlock, OffsetScaling
from omlt.io.keras import load_keras_sequential
#from omlt.neuralnet import FullSpaceSmoothNNFormulation
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

m.compressor_Mix_T = OmltBlock()
m.compressor_Mix_E = OmltBlock()

m.valve_vap_co = OmltBlock()
m.valve_liq_ch3oh_h2o = OmltBlock()
m.valve_liq_co2 = OmltBlock()
m.valve_T = OmltBlock()

m.dis_vap_co2= OmltBlock()
m.dis_vap_ch3oh = OmltBlock()
m.dis_vap_co=OmltBlock()
m.dis_vap_h2o=OmltBlock()
m.dis_con_T=OmltBlock()
m.dis_reb_T=OmltBlock()
m.dis_con_duty=OmltBlock()
m.dis_reb_duty=OmltBlock()
m.dis_no_stage=OmltBlock()
m.dis_feed_stage=OmltBlock()

m.cooler_liq_co2_2=OmltBlock()
m.cooler_liq_ch3oh_2=OmltBlock()
m.cooler_liq_co_2=OmltBlock()
m.cooler_liq_h2o_2=OmltBlock()
m.cooler_power_2=OmltBlock()

#%%
#defining inputs and outputs
inputs = ['F1', 'T1', 'P1', 'PR']
inputs2 = ['F2', 'T2', 'P2', 'PR2']
inputs3 = ['F3', 'T3', 'P3', 'PR3']
outputT = ['T2']
outputE = ['E1']
inputs4 = ['F1CO2', 'F1T', 'F2H2', 'F2T', 'P']
inputs5 = ['S1CO2','S1H2', 'S1T','S2CO2','S2H2','S2CH3OH','S2CO','S2H2O', 'S2T', 'P']
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
nn_compressor_T = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\LP_compressor\\lp_compressor_T_revised.keras', compile=False)
nn_compressor_E = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\LP_compressor\\lp_compressor_P_revised.keras', compile=False)
nn_compressor_T_HP = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\HP_compressor_CO2\\hp_compressor_co2_T_final.keras', compile=False)
nn_compressor_E_HP = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\HP_compressor_CO2\\hp_compressor_co2_P_3.keras', compile=False)
nn_compressor_T_HP2 = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\HP_compressor_H2\\hp_compressor_h2_T.keras', compile=False)
nn_compressor_E_HP2 = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\HP_compressor_H2\\hp_compressor_h2_P_2.keras', compile=False)

nn_mixer_feed = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Mixer_Feed\\mixer_feed_T_4.keras', compile=False)
nn_mixer_reactor = keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Mixer_Recycle\\mixer_recycle_T_2.keras', compile=False)

nn_reactor_comp = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_major_comp_relu.keras', compile=False)
nn_reactor_ch3oh = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_ch3oh_relu.keras', compile=False)
nn_reactor_h2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_h2_relu.keras', compile=False)
nn_reactor_T = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\pfr_T_relu.keras', compile=False)
nn_reactor_P = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\reactor_pressure_ergun_relu.keras', compile=False) 

nn_cooler_vap_co2_co = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_vap_co2_co.keras', compile=False)  
nn_cooler_vap_h2 = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_vap_h2_relu.keras', compile=False) 
nn_cooler_liq_ch3oh_h2o = keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_liq_met_h2o_relu.keras', compile=False) 

nn_compressor_T_mix= keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\HP_compressor_Mix\\hp_comp_mix_T_2.keras', compile=False)
nn_compressor_E_mix= keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\HP_compressor_Mix\\hp_compressor_mix_P_2.keras', compile=False)

nn_valve_vap_co=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_vap_co_relu.keras', compile=False)
nn_valve_liq_ch3oh_h2o=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_liq_ch3oh_h2o_relu.keras', compile=False)
nn_valve_liq_co2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_liq_co2.keras', compile=False)
nn_valve_T=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\valve_T_relu.keras', compile=False)

nn_dis_vap_co2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_vap_co2.keras', compile=False)
nn_dis_vap_ch3oh=keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Distillation_column\\dis_vap_ch3oh_2.keras', compile=False)
nn_dis_vap_co=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_vap_co.keras', compile=False)
nn_dis_vap_h2o=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_vap_h2o_relu.keras', compile=False)
nn_dis_con_T=keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Distillation_column\\dis_cond_T.keras', compile=False)
nn_dis_reb_T=keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Distillation_column\\dis_reb_T.keras', compile=False)
nn_dis_con_duty=keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Distillation_column\\dis_condensor_duty_1.keras', compile=False)
nn_dis_reb_duty=keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Distillation_column\\dis_reb_duty.keras', compile=False)
nn_dis_no_stage=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_no_of_stage_relu.keras', compile=False)
nn_dis_feed_stage=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\dis_feed_stage_relu.keras', compile=False)

nn_cooler_liq_co2_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_co2_2_relu.keras', compile=False)
nn_cooler_liq_ch3oh_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_ch3oh_2_relu.keras', compile=False)
nn_cooler_liq_co_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_co_2_relu.keras', compile=False)
nn_cooler_liq_h2o_2=keras.models.load_model('C:\\Users\\SHREEYA\\Cheat sheets\\Pyomo_practice\\cooler_h2o_2_relu.keras', compile=False)
nn_cooler_power_2=keras.models.load_model('C:\\Users\\SHREEYA\\component_wise_models_revised\\Cooler_2\\cooler_2_E_5.keras', compile=False)

#%%
#offset scaling=(Actual input-Lower Bound)/(Upper Bound -Lower Bound) where Lower Bound=offset and Difference =scaling
x_offset={'F1': 1900, 'T1': 20, 'P1':90, 'PR':2}
x_factor={'F1': 200, 'T1': 20, 'P1':260, 'PR':2}
x_offset2={'F2': 1900, 'T2': 5, 'P2':800, 'PR2':2}
x_factor2={'F2': 200, 'T2': 45, 'P2':3200, 'PR2':2}
x_offset3={'F3': 5000, 'T3': 20, 'P3':2500, 'PR3':2}
x_factor3={'F3': 2000, 'T3': 10, 'P3':1000, 'PR3':2}
x_offset4={'F1CO2':1900, 'F1T':30, 'F2H2':5000, 'F2T':30, 'P':6000}
x_factor4={'F1CO2':200, 'F1T':170, 'F2H2':2000, 'F2T':170, 'P':3000}
x_offset5={'S1CO2':1900,'S1H2':5000, 'S1T':100,'S2CO2':3000,'S2H2':30000,'S2CH3OH':100,'S2CO':1000,'S2H2O':10, 'S2T':40, 'P':6000}
x_factor5={'S1CO2':200,'S1H2':2000, 'S1T':100,'S2CO2':7000,'S2H2':20000,'S2CH3OH':250,'S2CO':3000,'S2H2O':90, 'S2T':60, 'P':3000}
x_offset6={'S1CO2':2000,'S1H2':35000,'S1CH3OH':20,'S1CO':500,'S1H2O':5, 'S1T':200,'S1P':5000, 'V':30}
x_factor6={'S1CO2':8000,'S1H2':25000,'S1CH3OH':330,'S1CO':3500,'S1H2O':95, 'S1T':100,'S1P':5000, 'V':30}
x_offset7={'S1CO2':800,'S1H2':29000,'S1CH3OH':170,'S1CO':540,'S1H2O':480, 'S1T':200,'S1P':300, 'S1dT':150.8935546875}
x_factor7={'S1CO2':8200,'S1H2':29100,'S1CH3OH':3040,'S1CO':4660,'S1H2O':3270, 'S1T':150,'S1P':9700, 'S1dT':178.623046875}

x_offset8={'S1CO2':3000,'S1H2':30000,'S1CH3OH':100,'S1CO':1000,'S1H2O':10, 'T':27,'P':4000, 'PR':1.05}
x_factor8={'S1CO2':7000,'S1H2':20000,'S1CH3OH':250,'S1CO':3000,'S1H2O':90, 'T':28,'P':3000, 'PR':0.95}

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
scaled_lb3=np.array([0,0,0,0,0,0,0,0,0,0])
scaled_ub3=np.array([1,1,1,1,1,1,1,1,1,1])
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
m.compressor_T1.build_formulation(ReluComplementarityFormulation(netT))
netT2= load_keras_sequential(nn_compressor_T, scaling_object=scalerT, scaled_input_bounds=scaled_input_bounds)
m.compressor_T2.build_formulation(ReluComplementarityFormulation(netT2))
netE = load_keras_sequential(nn_compressor_E, scaling_object=scalerE, scaled_input_bounds=scaled_input_bounds)
m.compressor_E1.build_formulation(ReluComplementarityFormulation(netE))
netE2 = load_keras_sequential(nn_compressor_E, scaling_object=scalerE, scaled_input_bounds=scaled_input_bounds)
m.compressor_E2.build_formulation(ReluComplementarityFormulation(netE2))

netT3 = load_keras_sequential(nn_compressor_T_HP, scaling_object=scalerT2, scaled_input_bounds=scaled_input_bounds)
m.compressor_T3.build_formulation(ReluComplementarityFormulation(netT3))
netT4 = load_keras_sequential(nn_compressor_T_HP, scaling_object=scalerT2, scaled_input_bounds=scaled_input_bounds)
m.compressor_T4.build_formulation(ReluComplementarityFormulation(netT4))
netE3 = load_keras_sequential(nn_compressor_E_HP, scaling_object=scalerE2, scaled_input_bounds=scaled_input_bounds)
m.compressor_E3.build_formulation(ReluComplementarityFormulation(netE3))
netE4 = load_keras_sequential(nn_compressor_E_HP, scaling_object=scalerE2, scaled_input_bounds=scaled_input_bounds)
m.compressor_E4.build_formulation(ReluComplementarityFormulation(netE4))

netT5 = load_keras_sequential(nn_compressor_T_HP2, scaling_object=scalerT3, scaled_input_bounds=scaled_input_bounds)
m.compressor_H2_T.build_formulation(ReluComplementarityFormulation(netT5))
netE5 = load_keras_sequential(nn_compressor_E_HP2, scaling_object=scalerE3, scaled_input_bounds=scaled_input_bounds)
m.compressor_H2_E.build_formulation(ReluComplementarityFormulation(netE5))

netT6 = load_keras_sequential(nn_mixer_feed, scaling_object=scalerT4, scaled_input_bounds=scaled_input_bounds2)
m.mixer_feed.build_formulation(ReluComplementarityFormulation(netT6))

netT7 = load_keras_sequential(nn_mixer_reactor, scaling_object=scalerT5, scaled_input_bounds=scaled_input_bounds3)
m.mixer_reactor.build_formulation(ReluComplementarityFormulation(netT7))

netR1 = load_keras_sequential(nn_reactor_comp, scaling_object=scalerR1, scaled_input_bounds=scaled_input_bounds4)
m.reactor_comp.build_formulation(ReluComplementarityFormulation(netR1))
netR2 = load_keras_sequential(nn_reactor_ch3oh, scaling_object=scalerR2, scaled_input_bounds=scaled_input_bounds4)
m.reactor_ch3oh.build_formulation(ReluComplementarityFormulation(netR2))
netR3 = load_keras_sequential(nn_reactor_h2, scaling_object=scalerR3, scaled_input_bounds=scaled_input_bounds4)
m.reactor_h2.build_formulation(ReluComplementarityFormulation(netR3))
netR4 = load_keras_sequential(nn_reactor_T, scaling_object=scalerR4, scaled_input_bounds=scaled_input_bounds4)
m.reactor_T.build_formulation(ReluComplementarityFormulation(netR4))
netR5 = load_keras_sequential(nn_reactor_P, scaling_object=scalerR5, scaled_input_bounds=scaled_input_bounds4)
m.reactor_P.build_formulation(ReluComplementarityFormulation(netR5))

netC1 = load_keras_sequential(nn_cooler_vap_co2_co, scaling_object=scalerC1, scaled_input_bounds=scaled_input_bounds4)
m.cooler_vap_co2_co.build_formulation(ReluComplementarityFormulation(netC1))
netC3 = load_keras_sequential(nn_cooler_vap_h2, scaling_object=scalerC3, scaled_input_bounds=scaled_input_bounds4)
m.cooler_vap_h2.build_formulation(ReluComplementarityFormulation(netC3))
netC5 = load_keras_sequential(nn_cooler_liq_ch3oh_h2o, scaling_object=scalerC5, scaled_input_bounds=scaled_input_bounds4)
m.cooler_liq_ch3oh_h2o.build_formulation(ReluComplementarityFormulation(netC5))

netT8 = load_keras_sequential(nn_compressor_T_mix, scaling_object=scalerT6, scaled_input_bounds=scaled_input_bounds4)
m.compressor_Mix_T.build_formulation(ReluComplementarityFormulation(netT8))
netE8 = load_keras_sequential(nn_compressor_E_mix, scaling_object=scalerE6, scaled_input_bounds=scaled_input_bounds4)
m.compressor_Mix_E.build_formulation(ReluComplementarityFormulation(netE8))

#%%
netV3 = load_keras_sequential(nn_valve_vap_co, scaling_object=scalerV3, scaled_input_bounds=scaled_input_bounds4)
m.valve_vap_co.build_formulation(ReluComplementarityFormulation(netV3))
netV5 = load_keras_sequential(nn_valve_liq_ch3oh_h2o, scaling_object=scalerV5, scaled_input_bounds=scaled_input_bounds4)
m.valve_liq_ch3oh_h2o.build_formulation(ReluComplementarityFormulation(netV5))
netV6 = load_keras_sequential(nn_valve_liq_co2, scaling_object=scalerV6, scaled_input_bounds=scaled_input_bounds4)
m.valve_liq_co2.build_formulation(ReluComplementarityFormulation(netV6))
netV8 = load_keras_sequential(nn_valve_T, scaling_object=scalerV8, scaled_input_bounds=scaled_input_bounds4)
m.valve_T.build_formulation(ReluComplementarityFormulation(netV8))

netD1 = load_keras_sequential(nn_dis_vap_co2, scaling_object=scalerD1, scaled_input_bounds=scaled_input_bounds5)
m.dis_vap_co2.build_formulation(ReluComplementarityFormulation(netD1))
netD2 = load_keras_sequential(nn_dis_vap_ch3oh, scaling_object=scalerD2, scaled_input_bounds=scaled_input_bounds5)
m.dis_vap_ch3oh.build_formulation(ReluComplementarityFormulation(netD2))
netD3 = load_keras_sequential(nn_dis_vap_co, scaling_object=scalerD3, scaled_input_bounds=scaled_input_bounds5)
m.dis_vap_co.build_formulation(ReluComplementarityFormulation(netD3))
netD4 = load_keras_sequential(nn_dis_vap_h2o, scaling_object=scalerD4, scaled_input_bounds=scaled_input_bounds5)
m.dis_vap_h2o.build_formulation(ReluComplementarityFormulation(netD4))
netD5 = load_keras_sequential(nn_dis_con_T, scaling_object=scalerD5, scaled_input_bounds=scaled_input_bounds5)
m.dis_con_T.build_formulation(ReluComplementarityFormulation(netD5))
netD6 = load_keras_sequential(nn_dis_reb_T, scaling_object=scalerD6, scaled_input_bounds=scaled_input_bounds5)
m.dis_reb_T.build_formulation(ReluComplementarityFormulation(netD6))
netD7 = load_keras_sequential(nn_dis_con_duty, scaling_object=scalerD7, scaled_input_bounds=scaled_input_bounds5)
m.dis_con_duty.build_formulation(ReluComplementarityFormulation(netD7))
netD8 = load_keras_sequential(nn_dis_reb_duty, scaling_object=scalerD8, scaled_input_bounds=scaled_input_bounds5)
m.dis_reb_duty.build_formulation(ReluComplementarityFormulation(netD8))

netD12 = load_keras_sequential(nn_dis_no_stage, scaling_object=scalerD12, scaled_input_bounds=scaled_input_bounds5)
m.dis_no_stage.build_formulation(ReluComplementarityFormulation(netD12))
netD13 = load_keras_sequential(nn_dis_feed_stage, scaling_object=scalerD13, scaled_input_bounds=scaled_input_bounds5)
m.dis_feed_stage.build_formulation(ReluComplementarityFormulation(netD13))

netC01 = load_keras_sequential(nn_cooler_liq_co2_2, scaling_object=scalerC01, scaled_input_bounds=scaled_input_bounds4)
m.cooler_liq_co2_2.build_formulation(ReluComplementarityFormulation(netC01))
netC02 = load_keras_sequential(nn_cooler_liq_ch3oh_2, scaling_object=scalerC02, scaled_input_bounds=scaled_input_bounds4)
m.cooler_liq_ch3oh_2.build_formulation(ReluComplementarityFormulation(netC02))
netC03 = load_keras_sequential(nn_cooler_liq_co_2, scaling_object=scalerC03, scaled_input_bounds=scaled_input_bounds4)
m.cooler_liq_co_2.build_formulation(ReluComplementarityFormulation(netC03))
netC04 = load_keras_sequential(nn_cooler_liq_h2o_2, scaling_object=scalerC04, scaled_input_bounds=scaled_input_bounds4)
m.cooler_liq_h2o_2.build_formulation(ReluComplementarityFormulation(netC04))
netC05 = load_keras_sequential(nn_cooler_power_2, scaling_object=scalerC05, scaled_input_bounds=scaled_input_bounds4)
m.cooler_power_2.build_formulation(ReluComplementarityFormulation(netC05))

#%%

m.total_opex = pyo.Var(bounds=(5.5,7.1),initialize=6.8853429349333)
m.total_capex = pyo.Var(bounds=(10,13),initialize=12.673086529811417)

m.compc1=pyo.Var(bounds=(68,70),initialize=68.70681219137984)
m.compc2=pyo.Var(bounds=(60,62),initialize=60.787624739500906)
m.compc3=pyo.Var(bounds=(59,61),initialize=59.69669912517253)
m.compc4=pyo.Var(bounds=(58,60),initialize=58.771346475901986)
m.compc5=pyo.Var(bounds=(109,112),initialize=110.30817435958556)
m.compc6=pyo.Var(bounds=(146,149),initialize=147.6561743258424)

m.reactorc=pyo.Var(bounds=(50,52),initialize=50.65424761896861)

m.vesselc4=pyo.Var(bounds=(51,53),initialize=51.72707459459628)

m.TAC=pyo.Var(bounds=(0.415,0.46),initialize=0.45)
m.Total_Energy=pyo.Var(bounds=(20000,25000), initialize=23491.08049227524)
m.cooler_liq_ch3oh_2_out=pyo.Var(bounds=(1500,2000), initialize=1836.2128159302297)
m.Utility_Cost=pyo.Var(bounds=(0,10000), initialize=1337.18585884858)

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
m.con167 = pyo.Constraint(expr=m.compressor_H2_T.inputs[0] >= 5000)
m.con1671 = pyo.Constraint(expr=m.compressor_H2_T.inputs[0] <= 7000)
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

m.con157 = pyo.Constraint(expr=m.mixer_reactor.inputs[2]==m.mixer_feed.outputs[0])
#defining second mixer input2 #change this for recycle stream 
m.con158 = pyo.Constraint(expr=m.mixer_reactor.inputs[3]<=1.01*m.compressor_Mix_T.inputs[0])
m.con159 = pyo.Constraint(expr=m.mixer_reactor.inputs[4]<=1.01*m.compressor_Mix_T.inputs[1])
m.con160 = pyo.Constraint(expr=m.mixer_reactor.inputs[5]<=1.01*m.compressor_Mix_T.inputs[2])
m.con161 = pyo.Constraint(expr=m.mixer_reactor.inputs[6]<=1.01*m.compressor_Mix_T.inputs[3])
m.con162 = pyo.Constraint(expr=m.mixer_reactor.inputs[7]<=1.01*m.compressor_Mix_T.inputs[4])
m.con163 = pyo.Constraint(expr=m.mixer_reactor.inputs[8]<=1.01*m.compressor_Mix_T.outputs[0])

m.con1582 = pyo.Constraint(expr=m.mixer_reactor.inputs[3]>=0.99*m.compressor_Mix_T.inputs[0])
m.con1592 = pyo.Constraint(expr=m.mixer_reactor.inputs[4]>=0.99*m.compressor_Mix_T.inputs[1])
m.con1602 = pyo.Constraint(expr=m.mixer_reactor.inputs[5]>=0.99*m.compressor_Mix_T.inputs[2])
m.con1612 = pyo.Constraint(expr=m.mixer_reactor.inputs[6]>=0.99*m.compressor_Mix_T.inputs[3])
m.con1622 = pyo.Constraint(expr=m.mixer_reactor.inputs[7]>=0.99*m.compressor_Mix_T.inputs[4])
m.con1632 = pyo.Constraint(expr=m.mixer_reactor.inputs[8]>=0.99*m.compressor_Mix_T.outputs[0])

m.con170 = pyo.Constraint(expr=m.compressor_T4.inputs[2]*m.compressor_T4.inputs[3]==m.compressor_H2_T.inputs[2]*m.compressor_H2_T.inputs[3])
m.con171 = pyo.Constraint(expr=m.mixer_feed.inputs[4]>=7800) 
m.con1712 = pyo.Constraint(expr=m.mixer_feed.inputs[4]<=10000)
m.con172 = pyo.Constraint(expr=m.mixer_reactor.inputs[9]==m.mixer_feed.inputs[4])

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

m.con173 = pyo.Constraint(expr=m.reactor_comp.inputs[0]==m.mixer_reactor.inputs[0]+m.mixer_reactor.inputs[3])
m.con174 = pyo.Constraint(expr=m.reactor_comp.inputs[1]==m.mixer_reactor.inputs[1]+m.mixer_reactor.inputs[4])
m.con175 = pyo.Constraint(expr=m.reactor_comp.inputs[2]==m.mixer_reactor.inputs[5])
m.con176 = pyo.Constraint(expr=m.reactor_comp.inputs[3]==m.mixer_reactor.inputs[6])
m.con177 = pyo.Constraint(expr=m.reactor_comp.inputs[4]==m.mixer_reactor.inputs[7])
m.con178 = pyo.Constraint(expr=m.reactor_comp.inputs[6]==m.mixer_reactor.inputs[9])
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

m.con180 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[0]==m.reactor_comp.outputs[0]*3600)
m.con181 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[1]==m.reactor_h2.outputs[0]*3600)
m.con182 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[2]==m.reactor_ch3oh.outputs[0]*3600)
m.con183 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[3]==m.reactor_comp.outputs[1]*3600)
m.con184 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[4]==m.reactor_comp.outputs[2]*3600)
m.con185 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[5]==m.reactor_T.outputs[0])
m.con186 = pyo.Constraint(expr=m.cooler_vap_co2_co.inputs[6]==m.reactor_P.outputs[0]) 

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

#inputs to valve 
m.con272=pyo.Constraint(expr=m.valve_liq_co2.inputs[0] == m.cooler_vap_co2_co.inputs[0]-m.cooler_vap_co2_co.outputs[0]*3600)
m.con273=pyo.Constraint(expr=m.valve_liq_co2.inputs[1] == m.cooler_vap_co2_co.inputs[1]-m.cooler_vap_h2.outputs[0]*3600)
m.con274=pyo.Constraint(expr=m.valve_liq_co2.inputs[2] == m.cooler_liq_ch3oh_h2o.outputs[0]*3600)
m.con275=pyo.Constraint(expr=m.valve_liq_co2.inputs[3] == m.cooler_vap_co2_co.inputs[3]-m.cooler_vap_co2_co.outputs[1]*3600)
m.con276=pyo.Constraint(expr=m.valve_liq_co2.inputs[4] == m.cooler_liq_ch3oh_h2o.outputs[1]*3600)
m.con277=pyo.Constraint(expr=m.valve_liq_co2.inputs[5] == m.cooler_vap_co2_co.inputs[5]-m.cooler_vap_co2_co.inputs[7])
m.con278=pyo.Constraint(expr=m.valve_liq_co2.inputs[6] == m.cooler_vap_co2_co.inputs[6]-100) 
m.con300=pyo.Constraint(expr=(m.valve_liq_co2.inputs[6]-m.valve_liq_co2.inputs[7])>=101) 
m.con3001=pyo.Constraint(expr=(m.valve_liq_co2.inputs[6]-m.valve_liq_co2.inputs[7])<=200)
m.con3002=pyo.Constraint(expr=(m.valve_liq_co2.inputs[2]-m.valve_liq_ch3oh_h2o.outputs[0]*3600)>=0)

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

m.con365=pyo.Constraint(expr=m.dis_reb_duty.inputs[0] == m.dis_no_stage.inputs[0])
m.con366=pyo.Constraint(expr=m.dis_reb_duty.inputs[1] == m.dis_no_stage.inputs[1])
m.con367=pyo.Constraint(expr=m.dis_reb_duty.inputs[2] == m.dis_no_stage.inputs[2])
m.con368=pyo.Constraint(expr=m.dis_reb_duty.inputs[3] == m.dis_no_stage.inputs[3])
m.con369=pyo.Constraint(expr=m.dis_reb_duty.inputs[4] == m.dis_no_stage.inputs[4])
m.con370=pyo.Constraint(expr=m.dis_reb_duty.inputs[5] == m.dis_no_stage.inputs[5])
m.con371=pyo.Constraint(expr=m.dis_reb_duty.inputs[6] == m.dis_no_stage.inputs[6])

m.con397=pyo.Constraint(expr=m.dis_vap_co2.inputs[0] == m.valve_liq_co2.outputs[0]*3600)
m.con398=pyo.Constraint(expr=m.dis_vap_co2.inputs[1] == 0)
m.con399=pyo.Constraint(expr=m.dis_vap_co2.inputs[2] == m.valve_liq_ch3oh_h2o.outputs[0]*3600)
m.con400=pyo.Constraint(expr=m.dis_vap_co2.inputs[3] == m.valve_liq_co2.inputs[3]-m.valve_vap_co.outputs[0]*3600)
m.con401=pyo.Constraint(expr=m.dis_vap_co2.inputs[4] == m.valve_liq_ch3oh_h2o.outputs[1]*3600)
m.con402=pyo.Constraint(expr=m.dis_vap_co2.inputs[5] >= 70)
m.con403=pyo.Constraint(expr=m.dis_vap_co2.inputs[6] == m.valve_liq_co2.inputs[6]-m.valve_liq_co2.inputs[7]-10)
m.con404=pyo.Constraint(expr=m.dis_vap_ch3oh.outputs[0] <= m.valve_liq_ch3oh_h2o.outputs[0]*3600) 
m.con405=pyo.Constraint(expr=m.dis_vap_co2.outputs[0]*3600 <= m.valve_liq_co2.outputs[0]*3600)
m.con406=pyo.Constraint(expr=m.dis_vap_co.outputs[0]*3600 <= m.valve_liq_co2.inputs[3]-m.valve_vap_co.outputs[0]*3600)

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
m.capex_con=pyo.Constraint(expr=m.total_capex==600000/(8000*3600)*(m.compc1+m.compc2+m.compc3+m.compc4+m.compc5+m.compc6+m.reactorc+m.vesselc4))
m.totalE_con=pyo.Constraint(expr=m.Total_Energy==m.compressor_E1.outputs[0]+m.compressor_E2.outputs[0]+m.compressor_E3.outputs[0]+m.compressor_E4.outputs[0]+m.compressor_H2_E.outputs[0]+m.compressor_Mix_E.outputs[0])
m.opex_con=pyo.Constraint(expr=m.total_opex == (2.0722*(m.compressor_E1.inputs[0]*44*0.045+m.compressor_H2_E.inputs[0]*2*0.473)+m.Total_Energy*0.25)/3600)
m.check14=pyo.Constraint(expr=m.cooler_liq_ch3oh_2_out == m.cooler_liq_ch3oh_2.outputs[0])                          
m.TAC_con=pyo.Constraint(expr=m.TAC==(m.total_capex/100+m.total_opex+m.Utility_Cost/100000)/(m.cooler_liq_ch3oh_2_out*32/3600))

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
m.T_k= pyo.Var(m.K, bounds=[0,1000],initialize={0:568.125,1:523.150,2:413.0987,3:394.5512,4:392.4241,5:387.6938,6:348.8852,7:337.4949,8:313.0903,9:303.1500,10:258.1500})

#%%
m.THIN_link1=pyo.Constraint(expr=m.TIN_H[1]==m.compressor_T1.outputs[0]+273.15)
m.THIN_link2=pyo.Constraint(expr=m.TIN_H[2]==m.compressor_T2.outputs[0]+273.15)
m.THIN_link3=pyo.Constraint(expr=m.TIN_H[3]==m.compressor_T3.outputs[0]+273.15)
m.THIN_link4=pyo.Constraint(expr=m.TIN_H[4]==m.cooler_vap_co2_co.inputs[5]+273.15)
m.THIN_link5=pyo.Constraint(expr=m.TIN_H[5]==m.dis_con_T.outputs[0]+273.15)
m.THIN_link6=pyo.Constraint(expr=m.TIN_H[6]==m.dis_con_T.outputs[0]+273.15)

m.THOUT_link1=pyo.Constraint(expr=m.TOUT_H[1]==m.compressor_E2.inputs[1]+273.15)
m.THOUT_link2=pyo.Constraint(expr=m.TOUT_H[2]==m.compressor_E3.inputs[1]+273.15)
m.THOUT_link3=pyo.Constraint(expr=m.TOUT_H[3]==m.compressor_E4.inputs[1]+273.15)
m.THOUT_link4=pyo.Constraint(expr=m.TOUT_H[4]==m.compressor_Mix_T.inputs[5]+273.15)
m.THOUT_link5=pyo.Constraint(expr=m.TOUT_H[5]==m.cooler_liq_ch3oh_2.inputs[7]+273.15)
m.THOUT_link6=pyo.Constraint(expr=m.TOUT_H[6]==m.TIN_H[6]-1)

m.TCIN_link1=pyo.Constraint(expr=m.TIN_C[1]==m.mixer_reactor.outputs[0]+273.15)
m.TCIN_link2=pyo.Constraint(expr=m.TIN_C[2]==m.valve_T.outputs[0]+273.15)
m.TCIN_link3=pyo.Constraint(expr=m.TIN_C[3]==m.dis_reb_T.outputs[0]+273.15)

m.TCOUT_link1=pyo.Constraint(expr=m.TOUT_C[1]==m.reactor_comp.inputs[5]+273.15)
m.TCOUT_link2=pyo.Constraint(expr=m.TOUT_C[2]==m.dis_vap_co2.inputs[5]+273.15)
m.TCOUT_link3=pyo.Constraint(expr=m.TOUT_C[3]==m.TIN_C[3]+1)
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
m.alpha_i=pyo.Param(m.I, initialize=600) #ub1
m.alpha_j=pyo.Param(m.J, initialize=600) #ub2
m.beta_i=pyo.Param(m.I, initialize=600) #ub3
m.beta_j=pyo.Param(m.J, initialize=600) #ub4

#%%
#CREATING DYNAMIC TEMPERATURE GRID 
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

m.R_k = pyo.Var(m.K, bounds=[0,100000000],initialize={0:0,1:24298.39961199215,2:26196.82957535138,3:26703.167176204523,4:26815.246571111165,5:8490.737923572124,6:3918.9067984410394,7:4704.508376989262,8:47288.79625644517,9:0,10:0})

#Bounds on heat #big m constraints 
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
m.F_H = pyo.Var(m.IP, bounds=fb5, initialize={1: 22.2472, 2: 22.6500, 3:25.3917, 4:540.2646 , 5: 818.5446, 6: 20945.768, 7:0})
m.F_C = pyo.Var(m.J, bounds=fb6, initialize={1: 535.2123, 2:541.5824  ,3:18680.895, 4:11496.946817413796, 5: 422.18057653555024})
#%%
m.FH_link1=pyo.Constraint(expr=m.F_H[1]==22.2472)
m.FH_link2=pyo.Constraint(expr=m.F_H[2]==22.6500)
m.FH_link3=pyo.Constraint(expr=m.F_H[3]==25.3917)
m.FH_link4=pyo.Constraint(expr=m.F_H[4]==(m.cooler_vap_co2_co.inputs[0]+m.cooler_vap_co2_co.inputs[1]+m.cooler_vap_co2_co.inputs[2]+m.cooler_vap_co2_co.inputs[3]+m.cooler_vap_co2_co.inputs[4])/3600*34.125)
m.FH_link5=pyo.Constraint(expr=m.F_H[5]==(m.cooler_liq_ch3oh_2.inputs[0]+m.cooler_liq_ch3oh_2.inputs[1]+m.cooler_liq_ch3oh_2.inputs[2]+m.cooler_liq_ch3oh_2.inputs[3]+m.cooler_liq_ch3oh_2.inputs[4])/3600*1526.38) #1,208.4444520913267149567282049517
m.FH_link6=pyo.Constraint(expr=m.F_H[6]==m.dis_con_duty.outputs[0]) 

m.FC_link1=pyo.Constraint(expr=m.F_C[1]==(m.reactor_comp.inputs[0]+m.reactor_comp.inputs[1]+m.reactor_comp.inputs[2]+m.reactor_comp.inputs[3]+m.reactor_comp.inputs[4])/3600*31.845)
m.FC_link2=pyo.Constraint(expr=m.F_C[2]==(m.dis_vap_co2.inputs[0]+m.dis_vap_co2.inputs[1]+m.dis_vap_co2.inputs[2]+m.dis_vap_co2.inputs[3]+m.dis_vap_co2.inputs[4])/3600*508.775) #666.61
m.FC_link3=pyo.Constraint(expr=m.F_C[3]==m.dis_reb_duty.outputs[0])#directly equate ot to predicted duty 
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
    return m.Utility_Cost==sum(m.mu_H[i]*sum(m.QH_ik[i,k] for k in m.KI) for i in m.IHU)+sum(m.mu_C[j]*sum(m.QC_jk[j,k] for k in m.KI) for j in m.JCU)
m.obj_func = pyo.Constraint(rule=obj_pinch)
#%%
solver = pyo.SolverFactory('scip')
solver.options ={'limits/time': 21600, 'limits/gap': 0.01, 'numerics/feastol': 1e-4, 'numerics/dualfeastol': 1e-4, 'propagating/obbt/dualfeastol': 1e-04,'heuristics/multistart/maxboundsize': 55000}
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
print('F1T1:', pyo.value(m.mixer_reactor.inputs[2]))
print('F2CO2:', pyo.value(m.mixer_reactor.inputs[3]))
print('F2H2:', pyo.value(m.mixer_reactor.inputs[4]))
print('F2CH3OH:', pyo.value(m.mixer_reactor.inputs[5]))
print('F2CO:', pyo.value(m.mixer_reactor.inputs[6]))
print('F2H2O:', pyo.value(m.mixer_reactor.inputs[7]))
print('F2T1:', pyo.value(m.mixer_reactor.inputs[8]))
print('P1:', pyo.value(m.mixer_reactor.inputs[9]))
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
print('F1CO2:', pyo.value(m.valve_liq_co2.inputs[0]))
print('F1H2:', pyo.value(m.valve_liq_co2.inputs[1]))
print('F1CH3OH:', pyo.value(m.valve_liq_co2.inputs[2]))
print('F1CO:', pyo.value(m.valve_liq_co2.inputs[3]))
print('F1H2O:', pyo.value(m.valve_liq_co2.inputs[4]))
print('T1:', pyo.value(m.valve_liq_co2.inputs[5]))
print('P1:', pyo.value(m.valve_liq_co2.inputs[6]))
print('dP:', pyo.value(m.valve_liq_co2.inputs[7]))

print('VF2CO:', pyo.value(m.valve_vap_co.outputs[0]*3600))
print('LF2CO2:', pyo.value(m.valve_liq_co2.outputs[0]*3600))
print('LF2H2:  0')
print('LF2CH3OH:', pyo.value(m.valve_liq_ch3oh_h2o.outputs[0]*3600))
print('LF2H2O:', pyo.value(m.valve_liq_ch3oh_h2o.outputs[1]*3600))
print('T2:', pyo.value(m.valve_T.outputs[0]))

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

#%%
print('Utilities_cost: ', pyo.value(m.Utility_Cost))
#%%
print('Tk0: ', pyo.value(m.T_k[0]))
print('Tk1: ', pyo.value(m.T_k[1]))
print('Tk2: ', pyo.value(m.T_k[2]))
print('Tk3: ', pyo.value(m.T_k[3]))
print('Tk4: ', pyo.value(m.T_k[4]))
print('Tk5: ', pyo.value(m.T_k[5]))
print('Tk6: ', pyo.value(m.T_k[6]))
print('Tk7: ', pyo.value(m.T_k[7]))
print('Tk8: ', pyo.value(m.T_k[8]))
print('Tk9: ', pyo.value(m.T_k[9]))
print('Tk10: ', pyo.value(m.T_k[10]))
#%%
print(pyo.value(m.MAT))
#%%
print('Rk0: ', pyo.value(m.R_k[0]))
print('Rk1: ', pyo.value(m.R_k[1]))
print('Rk2: ', pyo.value(m.R_k[2]))
print('Rk3: ', pyo.value(m.R_k[3]))
print('Rk4: ', pyo.value(m.R_k[4]))
print('Rk5: ', pyo.value(m.R_k[5]))
print('Rk6: ', pyo.value(m.R_k[6]))
print('Rk7: ', pyo.value(m.R_k[7]))
print('Rk8: ', pyo.value(m.R_k[8]))
print('Rk9: ', pyo.value(m.R_k[9]))
print('Rk10: ', pyo.value(m.R_k[10]))
#%%
print("Solution for variable Q_H:")
for i in m.I:
    for k in m.KI:
         print(f"Q_H[{i},{k}] = {m.QH_ik[i, k].value}")
         
#%%
print("Solution for variable Q_H1:")
for i in m.I:
    for k in m.KI:
         print(f"Q_H1[{i},{k}] = {m.QH1_ik[i, k].value}")
    
#%%
print("Solution for variable Q_H2:")
for i in m.I:
    for k in m.KI:
         print(f"Q_H2[{i},{k}] = {m.QH2_ik[i, k].value}")
         
#%%
print("Solution for variable Q_C:")
for j in m.J:
    for k in m.KI:
         print(f"Q_C[{j},{k}] = {m.QC_jk[j, k].value}")
         
#%%
print("Solution for variable Q_C1:")
for j in m.J:
    for k in m.KI:
         print(f"Q_C1[{j},{k}] = {m.QC1_jk[j, k].value}")
    
#%%
print("Solution for variable Q_C2:")
for j in m.J:
    for k in m.KI:
         print(f"Q_C2[{j},{k}] = {m.QC2_jk[j, k].value}")
         
#%%
print('FH1: ', pyo.value(m.F_H[1]))
print('FH2: ', pyo.value(m.F_H[2]))
print('FH3: ', pyo.value(m.F_H[3]))
print('FH4: ', pyo.value(m.F_H[4]))
print('FH5: ', pyo.value(m.F_H[5]))
print('FH6: ', pyo.value(m.F_H[6]))
print('FH7: ', pyo.value(m.F_H[7]))
print('FC1: ', pyo.value(m.F_C[1]))
print('FC2: ', pyo.value(m.F_C[2]))
print('FC3: ', pyo.value(m.F_C[3]))
print('FC4: ', pyo.value(m.F_C[4]))
print('FC5: ', pyo.value(m.F_C[5]))
#%%
print('THIN1: ', pyo.value(m.TIN_H[1]))
print('THIN2: ', pyo.value(m.TIN_H[2]))
print('THIN3: ', pyo.value(m.TIN_H[3]))
print('THIN4: ', pyo.value(m.TIN_H[4]))
print('THIN5: ', pyo.value(m.TIN_H[5]))
print('THIN6: ', pyo.value(m.TIN_H[6]))
print('THIN7: ', pyo.value(m.TIN_H[7]))
print('THOUT1: ', pyo.value(m.TOUT_H[1]))
print('THOUT2: ', pyo.value(m.TOUT_H[2]))
print('THOUT3: ', pyo.value(m.TOUT_H[3]))
print('THOUT4: ', pyo.value(m.TOUT_H[4]))
print('THOUT5: ', pyo.value(m.TOUT_H[5]))
print('THOUT6: ', pyo.value(m.TOUT_H[6]))
print('THOUT7: ', pyo.value(m.TOUT_H[7]))
print('TCIN1: ', pyo.value(m.TIN_C[1]))
print('TCIN2: ', pyo.value(m.TIN_C[2]))
print('TCIN3: ', pyo.value(m.TIN_C[3]))
print('TCIN4: ', pyo.value(m.TIN_C[4]))
print('TCIN5: ', pyo.value(m.TIN_C[5]))
print('TCOUT1: ', pyo.value(m.TOUT_C[1]))
print('TCOUT2: ', pyo.value(m.TOUT_C[2]))
print('TCOUT3: ', pyo.value(m.TOUT_C[3]))
print('TCOUT4: ', pyo.value(m.TOUT_C[4]))
print('TCOUT5: ', pyo.value(m.TOUT_C[5]))

#%%
print("Solution for variable TIN_HD:")
for i in m.I:
    for k in m.K:
         print(f"TIN_HD[{i},{k}] = {m.TIN_HD[i, k].value}")
#%%        
print("Solution for variable TOUT_HD:")
for i in m.I:
    for k in m.KI:
         print(f"TOUT_HD[{i},{k}] = {m.TOUT_HD[i, k].value}")
         
#%%
print("Solution for variable TIN_CD:")
for j in m.J:
    for k in m.K:
         print(f"TIN_CD[{j},{k}] = {m.TIN_CD[j, k].value}")
#%%        
print("Solution for variable TOUT_CD:")
for j in m.J:
    for k in m.KI:
         print(f"TOUT_CD[{j},{k}] = {m.TOUT_CD[j, k].value}")
#%%
print("Solution for variable XH_ik:")
for i in m.I:
    for k in m.K:
         print(f"XH_ik[{i},{k}] = {m.XH_ik[i, k].value}")
       
print("Solution for variable YH_ik:")
for i in m.I:
    for k in m.KI:
         print(f"YH_ik[{i},{k}] = {m.YH_ik[i, k].value}")
         
print("Solution for variable ZH_ik:")
for i in m.I:
    for k in m.KI:
         print(f"ZH_ik[{i},{k}] = {m.ZH_ik[i, k].value}")
         
#%%
print("Solution for variable XC_jk:")
for j in m.J:
    for k in m.K:
         print(f"XC_jk[{j},{k}] = {m.XC_jk[j, k].value}")
       
print("Solution for variable YC_jk:")
for j in m.J:
    for k in m.KI:
         print(f"YC_jk[{j},{k}] = {m.YC_jk[j, k].value}")
         
print("Solution for variable ZC_jk:")
for j in m.J:
    for k in m.KI:
         print(f"ZC_jk[{j},{k}] = {m.ZC_jk[j, k].value}")