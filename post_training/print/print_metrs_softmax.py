import numpy as np
import util
import pandas as pd

class PrintMetrsSoftmax:

    @staticmethod
    def in_sheet(f1_scores): 
        field_names = _create_fields_sheet()
        sheet_values = _put_values_sheet(f1_scores)
       
        df = pd.DataFrame(sheet_values, columns=field_names)
        df.to_excel(util.SHEET.PRINT_PATH_SOFT, sheet_name="Compare Softmax x Prob", index=False)
        
    
def _create_fields_sheet():
    list = []
    
    for fold in util.FOLDS:
        for metr in util.TABLE.METRICS_SOFT:
            item = f"FOLD {fold}: {metr.upper()} (%)"
            list.append(item)  
            
    field_names = util.SHEET.FIELD_NAMES 
    field_names.extend(list)
    
    return field_names

def _put_values_sheet(f1_scores):

    if util.DO_EARLY_FUSION:
        specs_fusions = util.SPECTRUMS + [util.EARLY_FUSION]
    else:
        specs_fusions = util.SPECTRUMS
    
    sheet_values = []

    models = util.MODELS
    folds = util.FOLDS

    for model in models:
        for spec in specs_fusions:
            data = [model.upper(), spec.upper()]
            metrs_list = []
            for fold in folds: 
                cont = models.index(model) * len(folds) * len(specs_fusions) + folds.index(fold) * len(specs_fusions) + specs_fusions.index(spec)
                f1_soft_unit, f1_prob_unit = f1_scores[cont]
                metrs = [f1_soft_unit, f1_prob_unit]
                metrs_list.extend(metrs)
    
            data.extend(metrs_list)
            comma_data = [str(number).replace('.', ',') for number in data]
            sheet_values.append(comma_data)    
    
    return sheet_values    