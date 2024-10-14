import numpy as np
import util
import pandas as pd

class PrintPerformanceSplit:

    @staticmethod
    def in_sheet(objs): 
        field_names = _create_fields_sheet()
        sheet_values = _put_values_sheet(objs)
       
        df = pd.DataFrame(sheet_values, columns=field_names)
        df.to_excel(util.SHEET.PRINT_PATH, sheet_name="util.SHEET.NAME", index=False)
        
    
def _create_fields_sheet():
    list = []
    
    for metr in util.TABLE.METRICS:
        item = f"{metr.upper()}(%)"
        list.append(item)  
            
    field_names = util.SHEET.FIELD_NAMES 
    field_names.extend(list)
    
    return field_names

def _put_values_sheet(objs):
    if util.USING_LATE_FUSION:
        specs_fusions = util.SPECTRUMS + [util.CHOSEN_FUSION] + util.LATE_FUSIONS
    else:
        specs_fusions = util.SPECTRUMS + [util.CHOSEN_FUSION]
        
    sorted_specs = {spec.upper(): i for i, spec in enumerate(specs_fusions)}
    sorted_objs = sorted(objs, key=lambda obj: (-ord(obj.model[0]), sorted_specs[obj.spectrum.upper()]))
    
    sheet_values = []

    for obj in sorted_objs:
        name = f"{obj.spectrum.upper()}"
        data = [obj.model.upper(), name]
        
        metrs_list = []
        f1s = obj.mean_f1s
        ious = obj.mean_ious
        
        metrs = [round(f1s*100, 2), round(ious*100, 2)]
        metrs_list.extend(metrs)
        data.extend(metrs_list)
    
        comma_data = [str(number).replace('.', ',') for number in data]
        sheet_values.append(comma_data)    
    
    return sheet_values    