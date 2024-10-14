from image import Image
from late_fusion  import LateFusion
import util

class Spectrum:
    def inputs_masks_preds(model, fold):
        inputs, masks, preds = [], [], []
        inputs_list = util.SPECTRUMS.copy()
        masks_list = util.SPECTRUMS.copy()
        preds_list = util.SPECTRUMS.copy()
        
        if util.DO_EARLY_FUSION:
            inputs_list.append(util.EARLY_FUSION.lower())
            masks_list.append(util.EARLY_FUSION.lower())
            preds_list.append(util.EARLY_FUSION.lower())
        
        for input, mask, pred in zip(inputs_list, masks_list, preds_list):
            inputs.append(Image.from_file(name=util.INPUT, model=model, spectrum=input, fold=fold))
            masks.append(Image.from_file(name=util.MASK, model=model, spectrum=mask, fold=fold))
            preds.append(Image.from_file(name=util.PRED, model=model, spectrum=pred, fold=fold))
            
        return inputs, masks, preds
        
      
def _lateFusions(mask, preds):
        later_fusions = []
        later_fusions.append(LateFusion.simple_mean(preds=preds))
        later_fusions.append(LateFusion.weighted_mean(preds=preds, mask=mask))
        
        return later_fusions


    

    


