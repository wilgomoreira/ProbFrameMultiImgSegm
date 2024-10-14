import torch
from tqdm import tqdm
import os
import torch.nn as nn
import util

class Epoch:
    
    def run(args, file, spectrum_name, sel_set, model, opt_sched): 
        k_folds = util.K_FOLDS
        
        if util.TRAIN_MODEL:
            if k_folds is None:
                Epoch.train_test(args=args, file=file, spectrum_name=spectrum_name, 
                                 sel_set=sel_set, model=model, opt_sched=opt_sched)
            else:
                fold_dataloaders = sel_set
                Epoch.train_test_kfold(args=args, file=file, spectrum_name=spectrum_name, 
                                       fold_dataloaders=fold_dataloaders, model=model, opt_sched=opt_sched)
        else:
            torch_load = torch.load(file.best_model_file, 
                                    map_location=torch.device(util.MAPA_LOC_DEVICE))
            model.load_state_dict(torch_load)  
            
        
    def train(epo, args, model, train_dataloader, optimizer, scheduler, spectrum_name, device):
        lr_this_epo = util.LR_START
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        
        loss_fn = nn.BCEWithLogitsLoss()
        loss_avg = 0.
        model.train()
        
        for it, data in enumerate(train_dataloader):
            rgb, ndvi, gndvi, mask, id = data
            rgb, ndvi, gndvi, mask = util.FUNC.SHAPE_GPU.evaluate(args=args, rgb=rgb, ndvi=ndvi, 
                                                                    gndvi=gndvi, mask=mask)
            rgb, ndvi, gndvi, mask = rgb.to(device), ndvi.to(device), gndvi.to(device), mask.to(device)
            optimizer.zero_grad()
            output = _choose_output_mode(spec_name=spectrum_name, model=model, 
                                         rgb=rgb, ndvi=ndvi, gndvi=gndvi)
            loss = loss_fn(output, mask)    
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_avg += float(loss)

    def test(args, test_dataloader, model, spectrum_name, device):
        accuracies, f1_scores, recalls, precisions, ious = [], [], [], [], []
        model.eval()
    
        with torch.no_grad():
            for it, data in enumerate(test_dataloader):
                rgb, ndvi, gndvi, mask, id = data   
                rgb, ndvi, gndvi, mask = util.FUNC.SHAPE_GPU.evaluate(args=args, rgb=rgb, ndvi=ndvi, 
                                                               gndvi=gndvi, mask=mask)
                rgb, ndvi, gndvi, mask = rgb.to(device), ndvi.to(device), gndvi.to(device), mask.to(device)
                output = _choose_output_mode(spec_name=spectrum_name, model=model, 
                                             rgb=rgb, ndvi=ndvi, gndvi=gndvi)
                    
                probs = torch.sigmoid(output)
                preds = util.FUNC.DECISION.take_a_one(probs=probs)
                metrs = util.FUNC.METRIC.compute_performance_metrics(preds=preds, mask=mask)
                acc, f1_score, iou, precision, recall = metrs
                accuracies.append(acc)
                f1_scores.append(f1_score)
                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)
                
        samples_metrs = accuracies, f1_scores, ious, precisions, recalls    
        mean_metr_perc = util.FUNC.METRIC.mean_metrics_perc(metrs=samples_metrs)
        mean_acc_perc, mean_precis_perc, mean_recall_perc, mean_f1_scor_perc, mean_iou_perc =  mean_metr_perc

        return mean_f1_scor_perc
         
    def train_test(args, file, spectrum_name, sel_set, model, opt_sched):
        best_f1 = 0
        
        device = util.MAPA_LOC_DEVICE
        model.to(device)
        
        for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
            Epoch.train(epo=epo, args=args, model=model, 
                        train_dataloader=sel_set.train_dataloader, 
                        optimizer=opt_sched.optimizer, 
                        scheduler=opt_sched.scheduler, 
                        spectrum_name=spectrum_name, device=device)
            
            mean_f1_scor_perc = Epoch.test(args=args, test_dataloader=sel_set.test_dataloader, 
                                           model=model, spectrum_name=spectrum_name, device=device)
            
            if mean_f1_scor_perc > best_f1:
                best_f1 = mean_f1_scor_perc
                torch.save(model.state_dict(), file.best_model_files[0])
            
            torch.save(model.state_dict(), file.checkpoint_model_files[0])
            torch.save(opt_sched.optimizer.state_dict(), file.checkpoint_optim_files[0])

        torch_load = torch.load(file.best_model_files[0], 
                                map_location=torch.device(util.MAPA_LOC_DEVICE))
        
        model.load_state_dict(torch_load)  
          
        if os.path.exists(file.final_model_files[0]):
            os.remove(file.final_model_files[0])
  
        os.rename(file.checkpoint_model_files[0], file.final_model_files[0])
        
    def train_test_kfold(args, file, spectrum_name, fold_dataloaders, model, opt_sched):
        device = util.MAPA_LOC_DEVICE
        best_model_files = file.best_model_files
        
        for fold, (train_loader, test_loader) in enumerate(fold_dataloaders):
            print(f"Fold {fold + 1}")
            
            model.apply(reset_weights)  # Resetando o modelo para cada fold
            model.to(device)
            
            best_f1 = 0
            
            for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
                Epoch.train(epo=epo, args=args, model=model, 
                            train_dataloader=train_loader, 
                            optimizer=opt_sched.optimizer, 
                            scheduler=opt_sched.scheduler, 
                            spectrum_name=spectrum_name, device=device)

                mean_f1_scor_perc = Epoch.test(args=args, test_dataloader=test_loader, 
                                               model=model, spectrum_name=spectrum_name, device=device)

                if mean_f1_scor_perc > best_f1:
                    best_f1 = mean_f1_scor_perc
                    torch.save(model.state_dict(), best_model_files[fold])
                 

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Support Functions -----------------------------------------------------------------

def _choose_output_mode(spec_name, model, rgb, ndvi, gndvi):
    
    match spec_name:        
        case util.RGB:
            output, _ = model(rgb)
        
        case util.NDVI:
            output, _ = model(ndvi)  
            
        case util.GNDVI:
            output, _ = model(gndvi) 
        
        case util.EARLY_FUSION:
            image_fusion = torch.cat([rgb, ndvi, gndvi], dim=1) 
            output, _ = model(image_fusion)
    
    return output
