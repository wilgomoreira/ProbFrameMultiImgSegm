import util
import numpy as np

class Print():
    
    @staticmethod
    def screen_and_file(param, metr):
        mean_metr_perc = _mean_metrics_perc_print(metrs=metr)
        mean_acc_perc, mean_precis_perc, mean_recall_perc, mean_f1_scor_perc, mean_iou_perc = mean_metr_perc
        
        content =  (f"| - SAVED MODEL - {param.spectrum_name} - Model {param.model_name}" +
                    f" - Acc: {mean_acc_perc}%. Prec: {mean_precis_perc}%." +
                    f" Recall: {mean_recall_perc}%. F1: {mean_f1_scor_perc}%. IoU: {mean_iou_perc}%")   
        print(content)

        with open(util.SAVE.OVERAL_PERFM, 'a') as file:
            file.write(content)
            file.write('\n')   
            
# -------------SUPPORT FUNCTIONS------------------------

def _mean_metrics_perc_print(metrs):
    
    mean_accuracy = np.mean(metrs.accuracies)
    mean_f1_score = np.mean(metrs.f1_scores)
    mean_iou = np.mean(metrs.ious)
    mean_precision = np.mean(metrs.precisions)
    mean_recall = np.mean(metrs.recalls)

    mean_acc_perc = round(mean_accuracy * 100, util.ROUND_PERC)
    mean_f1_scor_perc = round(mean_f1_score * 100, util.ROUND_PERC)
    mean_iou_perc = round(mean_iou * 100, util.ROUND_PERC)
    mean_precis_perc = round(mean_precision * 100, util.ROUND_PERC)
    mean_recall_perc = round(mean_recall * 100, util.ROUND_PERC)

    return (mean_acc_perc, mean_precis_perc, mean_recall_perc, mean_f1_scor_perc, mean_iou_perc)
        
        
    

         