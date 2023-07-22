import mcmodules
import types
import keyparams
import numpy as np

#===========[ Cyanobacteria ]======================
def Module_LiquidWater_UAv1p0():
    global Module_LiquidWater
    ModuleTemp = mcmodules.Module()
    ModuleTemp.define_name('Liquid Water AE v1.0')
    ModuleTemp.add_input('Surface_Temperature')
    ModuleTemp.add_input('Surface_Pressure')
    ModuleTemp.add_output('Suitability')
    def _execute(self):
        global Surface_Pressure, Surface_Temperature, GSuitability
        #breakpoint()
        if keyparams.Surface_Pressure > 0.07 and keyparams.Surface_Temperature < 373. and keyparams.Surface_Temperature > 273.:
            keyparams.Suitability=1.0
        else:
            keyparams.Suitability=0.0
        print('Suitability calculated in module', keyparams.Suitability)
        # Key parameters are returned to the main function, allowing the Visualization
        # of the results. Below a dictionary is defined with the relevant values
        keyparams.runid = keyparams.runid+' | '+ 'Cyanobacteria AE V1.0'
        KeyParameters={
            'Suitability' : keyparams.Suitability,
            'Surface_Temperature' : np.ndarray.item(keyparams.Surface_Temperature),
            'Surface_Pressure' : np.ndarray.item(keyparams.Surface_Pressure),
            'Bond_Albedo' : np.ndarray.item(keyparams.Bond_Albedo),
            'GreenhouseWarming' : np.ndarray.item(keyparams.GreenhouseWarming),
            'runid' : keyparams.runid+' | '+ 'Cyanobacteria AE V1.0'
            }
        return KeyParameters

    ModuleTemp.execute = types.MethodType(_execute, ModuleTemp)
    ModuleTemp.activate()
    #ModuleTemp.define_ID(m_id)
    Module_LiquidWater = ModuleTemp
    return Module_LiquidWater