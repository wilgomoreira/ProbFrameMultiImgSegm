import util
from logit_prob_estimator import LogitProbEstimator

class RefinePreds:
    
    @staticmethod
    def apply(all_images):
        inputs, masks, preds = all_images
        
        if util.LOG_PROB_ESTIMATOR:
            preds = LogitProbEstimator.evaluate(inputs=inputs, preds=preds, masks=masks)
            
        return preds
    
