# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:01:37 2020

@author: ASUS NT
"""

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from brukeropusreader import read_file

'''Measurement Class'''


class Measurement:

    def __init__(self, ZeroList, FieldList, units='cm-1', B=None, OPUS=False, limits=(None, None)):

        self.ZeroList = ZeroList  # [0][:]
        self.FieldList = FieldList  # [0][:]
        # self.path = path
        self.units = units
        self.OPUS = OPUS
        self.limits = limits
        if B is None:
            self.B = self.get_field()
        else:
            self.B = B

    '''Basic utilities'''

    # def file_list(self):
    # os.chdir(self.path)
    # return os.listdir()

    def get_field_legacy(self):
        # last version, do not work properly if you start at something else than zero field...
        maxField = float(self.FieldList[-1][-11:-5].replace('p', '.'))
        stepField = maxField / (np.size(self.FieldList))
        return np.arange(stepField, maxField + stepField, stepField)

    def get_field(self):
        # just read it from file names
        B = np.array([])
        for bv in self.FieldList:
            temp = float(bv[-11:-5].replace('p', '.'))
            B = np.append(B, temp)
        return B

    def set_units(self, data_ch):
        if self.units == 'cm-1':
            data_ch = data_ch.rename_axis('Energy (cm-1)')
            return data_ch
        elif self.units == 'meV':
            data_ch = data_ch.rename_axis('Energy (meV)')
            data_ch = data_ch.set_index(data_ch.index / 8.0656)
            return data_ch
        elif self.units == 'THz':
            data_ch = data_ch.rename_axis('Energy (THz)')
            data_ch = data_ch.set_index(data_ch.index / 33.35641)
            return data_ch
        # elif self.units== 'um': #TODO make it work !
        #     data_ch=data_ch.rename_axis('Energy (um)')
        #     data_ch=data_ch.set_index(data_ch.index/10000)
        #     return data_ch

        else:
            print('Wrong units parameter! cm-1/meV/THz only')

    def interpolate_macro(self, data_og):
        B = self.B
        Fint = interp1d([B[0], B[-1]], data_og.values)
        return pd.DataFrame(data=Fint(B), index=data_og.index,
                            columns=['{:1.2f}T'.format(B[i]) for i in range(0, np.size(B))])

    '''DATA'''

    def ZeroField(self):
        if self.OPUS == False:
            ZeroField = pd.read_csv(self.ZeroList[0], sep='\t', header=None, \
                                    index_col=0, names=['0Ta00T'])

            if len(self.ZeroList) == 2:
                ZeroField['0Ta{:1.2f}T'.format(self.B[-1])] = pd.read_csv(self.ZeroList[1], sep='\t', header=None,
                                                                          index_col=0)
                ZeroField = self.interpolate_macro(ZeroField)

        elif self.OPUS == True:
            ZeroField = pd.DataFrame(data=read_file(self.ZeroList[0])['ScSm'], \
                                     index=read_file(self.ZeroList[0]).get_range('ScSm'), columns=['0Ta00T'])

            if len(self.ZeroList) == 2:
                ZeroField['0Ta{:1.2f}T'.format(self.B[-1])] = pd.DataFrame(data=read_file(self.ZeroList[1])['ScSm'], \
                                                                           index=read_file(self.ZeroList[1]).get_range(
                                                                               'ScSm'))
                ZeroField = self.interpolate_macro(ZeroField)

        ZeroField = self.set_units(data_ch=ZeroField)

        return ZeroField.loc[self.limits[0]:self.limits[1]]

    def Field(self):
        B = self.B

        if self.OPUS == False:
            Field = pd.read_csv(self.FieldList[0], sep='\t', header=None, \
                                index_col=0, names=['{:1.2f}T'.format(B[0])])

            for i in range(1, np.size(self.FieldList)):
                Field['{:1.2f}T'.format(B[i])] = pd.read_csv(self.FieldList[i], sep='\t', \
                                                             header=None, index_col=0)
        elif self.OPUS == True:
            Field = pd.DataFrame(data=read_file(self.FieldList[0])['ScSm'], \
                                 index=read_file(self.FieldList[0]).get_range('ScSm'),
                                 columns=['{:1.2f}T'.format(B[0])])

            for i in range(1, np.size(self.FieldList)):
                Field['{:1.2f}T'.format(B[i])] = pd.DataFrame(data=read_file(self.FieldList[i])['ScSm'], \
                                                              index=read_file(self.FieldList[i]).get_range('ScSm'))

        Field = self.set_units(data_ch=Field)

        return Field.loc[self.limits[0]:self.limits[1]]

    def Ratio(self):
        # return self.Field()/self.ZeroField()
        if len(self.ZeroList) == 1:
            return self.Field().div(self.ZeroField().values, axis=0)
        elif len(self.ZeroList) > 1:
            return self.Field() / self.ZeroField()


'''Models'''

# class Models(object):
#     @staticmethod
#     def Dirac():
#         h = 1.054571800 * 10 ** -34  # Dirac constant in J*s
#         e = 1.602 * 10 ** -19  # elementary charge
#
#         E = np.array([(np.sqrt(2*e*h*B*n*(vf)**2+(delta*e)**2)*(1/e)\
#                        + np.sqrt(2*e*h*B*(n+1)*(vf)**2+(delta*e)**2)*(1/e))*1000 for B in B])


'''HF segment class'''

# class HF_Measurement(Measurement):
#
#     def __init__(self, path, B = None, units='cm-1'):
#         self.path = path
#         self.B = B
#         self.units = units
#
#     '''Segment'''
#     def Segment(self, N_m, B_s = None, Seg=None):
#
#         preList = self.file_list()
#
#         if Seg is None:
#             Seg_list = preList
#         else:
#             Seg_list = preList[Seg[0]:Seg[1]]
#
#         if B_s is None:
#             Seg_list = preList[Seg[0]:Seg[1]]
#
#         else:
#             B_s = self.B
#             B_s = B_s
#
#
#         fieldIndex = np.arange(0,N_m*np.size(B_s),N_m)
#
#         Run_avr = pd.DataFrame()
#
#         for i in range(0,np.size(B_s)):
#             run_N_avr = pd.DataFrame()
#             for j in range(0,N_m):
#                 run_N_avr[f'{j}'] = pd.Series(read_file(Seg_list[fieldIndex[i] + j])['ScSm'],
#                                              index = read_file(Seg_list[fieldIndex[i] + j]).get_range('ScSm'))
#             Run_avr[f'{B_s[i]:1.2f}T'] = run_N_avr.mean(axis=1)
#
#
#         Run_avr = self.set_units(data_ch=Run_avr)
#
#         return Run_avr
        

'''Treatment Class'''

class Treatment(object):
    
    @staticmethod
    def derivative(data,axis=0,edge=1):
        grad = np.gradient(data.values,edge,axis=axis)#[0]
        data_der_pd = pd.DataFrame(data=grad,index=data.index,columns=data.columns)
        return data_der_pd
    
    @staticmethod 
    def BS_correct(data,region):
        mean = np.mean(data.loc[region[0]:region[1]].values,axis=0) - 1
        return data - mean
    
    @staticmethod
    def Interpolate(data,B0,B):
        Fint = interp1d(B0,data.values,fill_value='extrapolate')
        return pd.DataFrame(data=Fint(B),index=data.index,
                            columns=['{:1.2f}T'.format(B[i]) for i in range(0,np.size(B))])
    
    @staticmethod
    def Change_units(data):
        if data.index.name == 'Energy (meV)':
            data=data.rename_axis('Energy (cm-1)')
            data=data.set_index(data.index*8.0656)
            print('Units changed from meV to cm-1')
            return data
        elif data.index.name == 'Energy (cm-1)':
            data=data.rename_axis('Energy (meV)')
            data=data.set_index(data.index/8.0656)    
            print('Units changed from cm-1 to meV')
            return data
        else:
            print('Cannot change units!')
            
    @staticmethod
    def SG_smooth(data,window,poly):
        data_s = pd.DataFrame(data=savgol_filter(data.iloc[:,0],window,poly),index=data.index,columns=[data.columns[0]])
        for i in range(1,np.size(data.columns)):
            data_s[data.columns[i]] = pd.Series(savgol_filter(data.iloc[:,i],window,poly),index=data_s.index)
        return data_s

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]