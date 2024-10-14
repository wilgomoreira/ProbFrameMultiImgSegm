import torch
from dataclasses import dataclass
import os
import util
import numpy as np

@dataclass
class Metric:
    accuracies: list
    f1_scores: list
    ious: list
    precisions: list
    recalls: list 
    
    @classmethod
    def evaluate(clc, args, file, spec_name, dataloader, model):
        
        all_acc_test, all_f1_test, all_ious_test, all_prec_test, all_rec_test = [], [], [], [], []
        
        if util.K_FOLDS != None:
            for fold, (train_loader, test_loader) in enumerate(dataloader):
                save_pred_mask_dir_train = file.save_pred_mask_dirs_train[fold]
                save_pred_mask_dir_test = file.save_pred_mask_dirs_test[fold]
                best_model_file = file.best_model_files[fold]
                
                torch_load = torch.load(best_model_file, 
                                        map_location=torch.device(util.MAPA_LOC_DEVICE))
            
                model.load_state_dict(torch_load)  
                
                model.train()
                _ , _ , _, _, _ = performance_for_each_image(args, save_pred_mask_dir_train, 
                                                             spec_name, train_loader, model)   
                model.eval()
                acc_test, f1_test, ious_test, prec_test, recalls_test = performance_for_each_image(args, 
                                                                                                    save_pred_mask_dir_test, 
                                                                                                    spec_name, 
                                                                                                    test_loader, 
                                                                                                    model)   
    
                all_acc_test.append(np.mean(acc_test))
                all_f1_test.append(np.mean(f1_test))
                all_ious_test.append(np.mean(ious_test))
                all_prec_test.append(np.mean(prec_test))
                all_rec_test.append(np.mean(recalls_test))
        
        else:
            train_loader = dataloader.train_dataloader
            test_loader = dataloader.test_dataloader
            
            save_pred_mask_dir_train = file.save_pred_mask_dirs_train[0]
            save_pred_mask_dir_test = file.save_pred_mask_dirs_test[0]
            best_model_file = file.best_model_files[0]
            
            torch_load = torch.load(best_model_file, map_location=torch.device(util.MAPA_LOC_DEVICE))
            model.load_state_dict(torch_load)  
            
            model.train()
            _ , _ , _, _, _ = performance_for_each_image(args, save_pred_mask_dir_train, 
                                                             spec_name, train_loader, model)   
            model.eval()
            acc_test, f1_test, ious_test, prec_test, recalls_test = performance_for_each_image(args, 
                                                                                                save_pred_mask_dir_test, 
                                                                                                spec_name, 
                                                                                                test_loader, 
                                                                                                model)   
            all_acc_test.append(np.mean(acc_test))
            all_f1_test.append(np.mean(f1_test))
            all_ious_test.append(np.mean(ious_test))
            all_prec_test.append(np.mean(prec_test))
            all_rec_test.append(np.mean(recalls_test))
            
        all_acc_test_np = np.array(all_acc_test)
        all_f1_test_np = np.array(all_f1_test)
        all_ious_test_np = np.array(all_ious_test)
        all_prec_test_np = np.array(all_prec_test)
        all_rec_test_np = np.array(all_rec_test)
            
        return clc(accuracies=all_acc_test_np, 
                   f1_scores=all_f1_test_np, 
                   ious=all_ious_test_np, 
                   precisions=all_prec_test_np, 
                   recalls=all_rec_test_np)
        
def performance_for_each_image(args, save_pred_mask_dir, spec_name, dataloader, model):    
    accuracies, f1_scores, recalls, precisions, ious = [], [], [], [], []  

    with torch.no_grad():
        for it, data in enumerate(dataloader):
            rgb, ndvi, gndvi, mask, id = data   
            rgb, ndvi, gndvi, mask = util.FUNC.SHAPE_GPU.evaluate(args=args, rgb=rgb, ndvi=ndvi, 
                                                                    gndvi=gndvi, mask=mask)
                
            input_mod = util.FUNC.DECISION.choose_input_mode(spec_name=spec_name, rgb=rgb, ndvi=ndvi, gndvi=gndvi)
            t0, t1, output = util.FUNC.DECISION.time_for_model(model=model, input_mod=input_mod)
            probs = torch.sigmoid(output)
            preds = util.FUNC.DECISION.take_a_one(probs=probs)
            metrs = util.FUNC.METRIC.compute_performance_metrics(preds=preds, mask=mask)
            
            acc, f1_score, iou, precision, recall = metrs
            accuracies.append(acc)
            f1_scores.append(f1_score)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            
            input_mod = input_mod.detach().cpu().numpy()
            mask = mask.cpu().detach().numpy()
            probs = probs.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            
            _for_post_training(input_mod=input_mod, mask=mask, probs=output, id=id, 
                                save_pred_mask_dir=save_pred_mask_dir)
    
    return (np.array(accuracies), np.array(f1_scores), 
            np.array(ious), np.array(precisions), np.array(recalls))


def _for_post_training(input_mod, mask, probs, id, save_pred_mask_dir):
    for i, (input, mask, pred, label) in enumerate(zip(input_mod, mask, probs, id)):
        # Convertendo os dados para arrays numpy
        input_np = np.array(input)
        mask_np = np.array(mask)
        pred_np = np.array(pred)
        
        # Salvando todos os componentes em um arquivo .npz compactado
        np.savez_compressed(os.path.join(save_pred_mask_dir, label + '.npz'), input=input_np, mask=mask_np, pred=pred_np)


