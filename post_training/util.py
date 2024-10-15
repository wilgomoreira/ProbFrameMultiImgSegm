import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

DEFAULT_SPEC = "rgb"
MODELS = ["segnet", "deeplab"]
SPECTRUMS = ["rgb", "ndvi", "gndvi"]
FOLDS = [1, 2, 3]
DEFAULT_FOLD = 1

IMG_HEIGHT = 240
IMG_WIDTH = 240

LOG_PROB_ESTIMATOR = True
PERF_METRIC = "ECE"    # F1_SCORE  AVER_PREC   ECE

INPUT, MASK, PRED = ("INPUT", "MASK", "PRED")
RGB, NDVI, GNDVI = ("rgb", "ndvi", "gndvi")
EARLY_FUSION = "EARLY_FUSION"
DO_EARLY_FUSION = True

# LATE FUSION    IT IS NOT WORKING
USING_LATE_FUSION = False
LATE_FUSION = "LATE_FUSION"
L_FUSION_SIMPLE_MEAN = "LATE_FUSION: SIMPLE_MEAN"
L_FUSION_WEIGHTED_MEAN = "LATE_FUSION: WEIGHTED_MEAN"
LATE_FUSIONS = [L_FUSION_SIMPLE_MEAN, L_FUSION_WEIGHTED_MEAN]

MET_OTSU = False
THRESHOLD = 0.5

#Calibration
CALIBRATION = False
CALIB_ISO_REGR = "iso_regression"   # use in likehoods
CALIB_TEMPERATURE = "temperature"   # use in logits
CALIB_MET = CALIB_ISO_REGR  

#Condicional Random Field (CRF)
CONDICIONAL_RANDOM_FIELD = False
PAIRWISEGAUS_SXY = 1  # initial:3       best:1
PAIRWISEGAUS_COMPAT = 1  # initial:3    best:1
PAIRBILATERAL_SXY = 20   # initial:80   best:20
PAIRBILATERAL_SRGB = 4  # initial:13   best:4
PAIRBILATERAL_COMPAT = 2 # initial:10  best:2
INFERENCE = 5   #iterations   

#BAYES
BAYES = False
MODEL_NAME_BAYES = "segnet_with_deeplab"
BAYES_WEIGHT1 = 0.85    #best:0.85
BAYES_WEIGHT2 = 0.15    #best:0.15
A_PRIOR_PROB = 0.5


class MET_LATE_FUSION:
    WEIGHT_F1S, WEIGHT_IOUS = ("f1s", "ious")

class DIR:
    PARENT = "./training/weights"
    GRANDSON_TRAIN = "pred_masks_train"
    GRANDSON_TEST = "pred_masks_test"
    EXTENSION = "*npz"
    
class SHEET:
    NAME = "RESULT SHEET"
    FIELD_NAMES = ["MODEL", "SPECTRUM"] 
    ORGANIZED_DATA = ("model", "spectrum")
    METRICS = ("f1s", "ious")
    
    PRINT_PATH = "post_training/print/res_perfomance_split.xlsx"
    PRINT_PATH_SOFT = "post_training/print/res_perfomance_soft.xlsx"
    
    METRC_NAME = "METRIC ECE"
    PRINT_PATH_ECE = "post_training/print/res_ece_metr.xlsx"
        
class TABLE:
    TITLE = "BEST RESULTS"
    FIELD_NAMES = ["MODEL", "SPECTRUM"] 
    MEAN_NAMES = ["MEAN_F1S(%)", "MEAN_IOUS(%)"]
    ITEMS_ALIGN = ["MODEL", "SPECTRUM"]
    LEFT, CENTER, RIGHT = ("l", "c", "r")
    CHOSEN_ALIGN = LEFT
    
    SORT = True
    METRICS = ("f1s", "ious")
    METRICS_SOFT = ("F1 SOFTMAX", "F1 PROB")
    S_F1S, S_IOUS = METRICS
    ITEM_SORT = S_F1S   
    ORGANIZED_DATA = ("name", "spectrum", "model")
    
    PRINT_PATH = "post_training/print/res_sorted_results.txt"
    
class PLOT:
    # Best Results
    PERMUT_IMG = (1,2,0)
    NOT_DECISION = "INPUT"
    
    CRITERIA_FOR_IMG = ("RANDOM", "MAX", "MIN", "SIMILAR_MEAN", "DEFINE")
    CRIT_RANDOM, CRIT_MAX, CRIT_MIN, CRIT_SIMILAR_MEAN, CRIT_DEFINE = CRITERIA_FOR_IMG
    CHOSEN_IMG = CRIT_RANDOM
    IMG_DEFINED = 109
    TOL_SIMILAR_MEAN = 0.01
    REVERSE = True 

    NOT_MODEL_NAME = ("INPUT", "MASK")
    INPUT, MASK = NOT_MODEL_NAME

    SUBPLOT_SIZE_BEST = (15,10)
    SUBTITLE_BEST = "INPUT, GROUND TRUTH AND PREDICTION"
    
    N_RESULTS = 3   #ROWS 
    N_COLUMNS = 3
    
    BEST_RESULT_PATH = "post_training/plot/res_plot_bestResults.png"
    
    # Comparative 
    COMPARATIVE_PATH = "post_training/plot/res_plot_comparative.png"
    SUBPLOT_SIZE_COMP = (25,10)
    SUBTITLE_COMP = "COMPARATIVE"
    DEFAULT_SPEC_COMP = RGB
    MODEL1_COMP = "segnet"
    DATA1_COMP = "t2"
    PRED1NAME_COMP = L_FUSION_WEIGHTED_MEAN
    SPECTRUM1_COMP = L_FUSION_WEIGHTED_MEAN
    MODEL2_COMP = "segnet"
    DATA2_COMP = "t2"
    PRED2NAME_COMP = L_FUSION_SIMPLE_MEAN
    SPECTRUM2_COMP = L_FUSION_SIMPLE_MEAN
    MODEL3_COMP = "segnet"
    DATA3_COMP = "t2"
    PRED3NAME_COMP = PRED
    SPECTRUM3_COMP = NDVI
    
    #Choose
    CHOOSE_ROOTS = { "true_image":'/media/wilgo/0610502510501E4D/greenaI_split/greenaI_split/FIELD/altum/masks/',
              "predict": "/home/wilgo/Downloads/MISAgro - Practing/training/weights/MODEL_tINDEX_BAND/pred_masks/"}

    CHOOSE_FILE_EXT = {'tif': '.tif', 'npy': '.npy', 'pickle': '.pickle'}

    CHOOSE_SUBPLOT_SIZE = (10,5)

    CHOOSE_SAVE_IMAGE = "post_training/plot/res_choosen_images.png"
    
    CHOOSE_IMAGES = { 'qbaixo': ['00480_04800.tif']}

    BANDS = ['RGB']             # (RGB, NDVI)
    MODELS = ['deeplab']       # (segnet, deeplab)


def sigmoid(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    y = torch.sigmoid(x)
    y_numpy = y.numpy()
    
    return y_numpy

def thresholding(probs):
    probs = np.array(probs)
    probs = torch.from_numpy(probs)
    bin_preds = (probs > THRESHOLD).float()
    
    return bin_preds

def perfomance_metrics(bin_preds, masks):
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
    
    return np.mean(f1s), np.mean(ious)


def compute_performance_metrics2(preds, mask):
    NOT_ZERO = 1e-7

    tp = torch.sum((preds == 1) & (mask == 1)).item()
    fp = torch.sum((preds == 1) & (mask == 0)).item()
    tn = torch.sum((preds == 0) & (mask == 0)).item()
    fn = torch.sum((preds == 0) & (mask == 1)).item()

    acc = (tp + tn) / (tp + fp + tn + fn + NOT_ZERO)
    precision = tp / (tp + fp + NOT_ZERO)
    recall = tp / (tp + fn + NOT_ZERO)
    f1_score = (2 * precision * recall) / (precision + recall + NOT_ZERO)

    intersection = torch.sum((preds == 1) & (mask == 1)).item()
    union = torch.sum((preds == 1) | (mask == 1)).item()
    iou = intersection / (union + NOT_ZERO)
    
    
    return np.mean(f1_score), np.mean(iou)



def calculate_roc_auc(pred_probs, true_labels):
    """
    pred_probs: previsões contínuas (valores de probabilidade)
    true_labels: rótulos binários (0 ou 1)
    """
    pred_probs = pred_probs.flatten()
    true_labels = true_labels.flatten()
    # Calcular FPR, TPR e thresholds
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    
    # Calcular a AUC
    roc_auc = auc(fpr, tpr)

    return roc_auc

def calculate_precision_recall_ap(pred_probs, true_labels):
    """
    pred_probs: previsões contínuas (valores de probabilidade ou scores)
    true_labels: rótulos binários (0 ou 1)
    """
    # Certifique-se de que ambos os arrays estão em formato unidimensional
    pred_probs = pred_probs.flatten()
    true_labels = true_labels.flatten()
    

    # Calcular Precisão e Recall para vários thresholds
    precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
    
    # Calcular a AP (Average Precision)
    ap_score = average_precision_score(true_labels, pred_probs)

    # Calcular a AUC da curva Precision-Recall (opcional)
    pr_auc = auc(recall, precision)

    return ap_score
    

    



    
    
    
   