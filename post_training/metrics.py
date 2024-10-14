from dataclasses import dataclass
import torch
import numpy as np
from image import Image
import util

@dataclass
class Metrics:
    name: str
    model: str
    spectrum: str
    
    f1s: np
    ious: np
    mean_f1s: np
    mean_ious: np
   
    preds: Image
    masks: Image
    inputs: Image
    
    @staticmethod
    def for_all_images(all_images):
        inputs, masks, preds = all_images
        all_metrs = []
        
        for input, mask, pred in zip(inputs, masks, preds):
            all_metrs.append(Metrics.evaluate(preds=pred, masks=mask, inputs=input))
         
        return all_metrs
    
    @classmethod   
    def evaluate(clc, preds, masks, inputs):
        f1s, ious = _calculate_metrics(preds=preds, masks=masks)
        mean_f1s, mean_ious = _mean_metrics(f1s=f1s, ious=ious)
        
        return clc(name=preds.name, model=preds.model, spectrum=preds.spectrum, 
                   f1s=f1s, ious=ious, mean_f1s=mean_f1s, mean_ious=mean_ious, 
                   inputs=inputs, masks=masks, preds=preds)
           

def _calculate_metrics(preds, masks):
    probs_train, probs_test = preds.value
    masks_train, masks_test = masks.value
    
    probs_test = util.sigmoid(probs_test)
    
    probs_test = util.thresholding(probs=probs_test)
    f1s, ious = _perfomance_metrics(bin_preds=probs_test, masks=masks_test)
        
    return f1s, ious
        
def _perfomance_metrics(bin_preds, masks):
    NOT_ZERO = 1e-7
    
    if isinstance(bin_preds, np.ndarray):
        bin_preds = torch.from_numpy(bin_preds)

    f1s, ious = [], []

    for bin_pred, mask in zip(bin_preds, masks):
        tp = torch.sum((bin_pred == 1) & (mask == 1)).item()
        fp = torch.sum((bin_pred == 1) & (mask == 0)).item()
        tn = torch.sum((bin_pred == 0) & (mask == 0)).item()
        fn = torch.sum((bin_pred == 0) & (mask == 1)).item()
        
        acc = (tp + tn) / (tp + fp + tn + fn + NOT_ZERO)
        precision = tp / (tp + fp + NOT_ZERO)
        recall = tp / (tp + fn + NOT_ZERO)
        f1_score = (2 * precision * recall) / (precision + recall + NOT_ZERO)

        intersection = torch.sum((bin_pred == 1) & (mask == 1)).item()
        union = torch.sum((bin_pred == 1) | (mask == 1)).item()
        iou = intersection / (union + NOT_ZERO)
        
        if acc == 0:
            acc = NOT_ZERO
        if f1_score == 0:
            f1_score = NOT_ZERO
        if iou == 0:
            iou = NOT_ZERO
        
        f1s.append(f1_score)
        ious.append(iou)
    
    f1s = np.array(f1s)
    ious = np.array(ious)   
    
    return f1s, ious

def _mean_metrics(f1s, ious):
    mean_f1s = np.mean(f1s)
    mean_ious = np.mean(ious) 
    
    return mean_f1s, mean_ious       


     



    
    
        
        
            
        
        

        
 