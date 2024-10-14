import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from sklearn.model_selection import KFold
from dataclasses import dataclass
import util

@dataclass
class SelectSet:
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    
    @classmethod
    def separate_them(cls, load_dat):
        batch_train = util.BATCH_SIZE_TRAIN
        batch_test = util.BATCH_SIZE_TEST
        shuffle_train = util.SUFFLE_TRAIN
        shuffle_test = util.SUFFLE_TEST
        k_folds = util.K_FOLDS

        if util.DATASET_SPLITING == util.HOLDOUT:
            fold_dataloaders = cls.holdout(load_dat, batch_train, batch_test, shuffle_train, shuffle_test)
        
        elif util.DATASET_SPLITING == util.CROSS_BYDOMAIN:
            fold_dataloaders = cls.crossByDomain(load_dat, batch_train, batch_test, shuffle_train, shuffle_test)
        
        elif util.DATASET_SPLITING == util.CROSS_BYFOLDS:
            fold_dataloaders = cls.crossByKfolds(load_dat, batch_train, batch_test, shuffle_train, shuffle_test, k_folds)
        
        return fold_dataloaders
    

    def holdout(load_dat, batch_train, batch_test, shuffle_train, shuffle_test):
        fold_dataloaders = []
        split_ratio = util.SPLIT_RATIO
        
        # Helper function to split dataset into train and test
        def split_dataset(dataset, split_ratio):
            train_size = int(split_ratio * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            return train_dataset, test_dataset
        
        # For loader_qbaixo
        train_dataset_qbaixo, test_dataset_qbaixo = split_dataset(load_dat.loader_qbaixo, split_ratio)
        train_dataloader_qbaixo = torch.utils.data.DataLoader(train_dataset_qbaixo, 
                                                            batch_size=batch_train, 
                                                            shuffle=shuffle_train, 
                                                            drop_last=True)
        test_dataloader_qbaixo = torch.utils.data.DataLoader(test_dataset_qbaixo, 
                                                            batch_size=batch_test, 
                                                            shuffle=shuffle_test,
                                                            drop_last=True)
        fold_dataloaders.append((train_dataloader_qbaixo, test_dataloader_qbaixo))
        
        # For loader_esac
        train_dataset_esac, test_dataset_esac = split_dataset(load_dat.loader_esac, split_ratio)
        train_dataloader_esac = torch.utils.data.DataLoader(train_dataset_esac, 
                                                            batch_size=batch_train, 
                                                            shuffle=shuffle_train,
                                                            drop_last=True)
        test_dataloader_esac = torch.utils.data.DataLoader(test_dataset_esac, 
                                                        batch_size=batch_test, 
                                                        shuffle=shuffle_test,
                                                        drop_last=True)
        fold_dataloaders.append((train_dataloader_esac, test_dataloader_esac))

        # For loader_val
        train_dataset_val, test_dataset_val = split_dataset(load_dat.loader_val, split_ratio)
        train_dataloader_val = torch.utils.data.DataLoader(train_dataset_val, 
                                                        batch_size=batch_train, 
                                                        shuffle=shuffle_train,
                                                        drop_last=True)
        test_dataloader_val = torch.utils.data.DataLoader(test_dataset_val, 
                                                        batch_size=batch_test, 
                                                        shuffle=shuffle_test,
                                                        drop_last=True)
        fold_dataloaders.append((train_dataloader_val, test_dataloader_val))

        return fold_dataloaders
    
    def crossByDomain(load_dat, batch_train, batch_test, shuffle_train, shuffle_test):
        fold_dataloaders = []
     
        # For T1
        train_dataset_t1 = torch.utils.data.ConcatDataset([load_dat.loader_val, 
                                                        load_dat.loader_esac]) 
        train_dataloader_t1 = torch.utils.data.DataLoader(train_dataset_t1, 
                                                        batch_size=batch_train, 
                                                        shuffle=shuffle_train)
        test_dataloader_t1 = torch.utils.data.DataLoader(load_dat.loader_qbaixo, 
                                                        batch_size=batch_test, 
                                                        shuffle=shuffle_test)

        fold_dataloaders.append((train_dataloader_t1, test_dataloader_t1))
        
         # For T2 
        train_dataset_t2 = torch.utils.data.ConcatDataset([load_dat.loader_val, 
                                                        load_dat.loader_qbaixo])
        train_dataloader_t2 = torch.utils.data.DataLoader(train_dataset_t2, 
                                                        batch_size=batch_train, 
                                                        shuffle=shuffle_train)
        test_dataloader_t2 = torch.utils.data.DataLoader(load_dat.loader_esac, 
                                                        batch_size=batch_test, 
                                                        shuffle=shuffle_test)  
        
        fold_dataloaders.append((train_dataloader_t2, test_dataloader_t2))            
           
        # For T3 
        train_dataset_t3 = torch.utils.data.ConcatDataset([load_dat.loader_esac, 
                                                        load_dat.loader_qbaixo])
        train_dataloader_t3 = torch.utils.data.DataLoader(train_dataset_t3, 
                                                        batch_size=batch_train, 
                                                        shuffle=shuffle_train)
        test_dataloader_t3 = torch.utils.data.DataLoader(load_dat.loader_val, 
                                                        batch_size=batch_test, 
                                                        shuffle=shuffle_test)

        fold_dataloaders.append((train_dataloader_t3, test_dataloader_t3)) 
        
        return fold_dataloaders
        
        
    def crossByKfolds(load_dat, batch_train, batch_test, shuffle_train, shuffle_test, k_folds):
        # Juntar todos os datasets
        all_datasets = ConcatDataset([load_dat.loader_val, load_dat.loader_esac, load_dat.loader_qbaixo])
        
        if k_folds is None:
            # Dividir em 70% treino e 30% teste
            train_size = int(0.70 * len(all_datasets))
            test_size = len(all_datasets) - train_size
            train_dataset, test_dataset = random_split(all_datasets, [train_size, test_size])
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_train, shuffle=shuffle_train)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_test, shuffle=shuffle_test)
            
            return train_dataloader, test_dataloader
        else:
            # Configuração para validação cruzada com K-fold
            kfold = KFold(n_splits=k_folds, shuffle=True)
            return create_kfold_dataloaders(all_datasets, kfold, batch_train, batch_test)


def create_kfold_dataloaders(dataset, kfold, batch_train, batch_test):
    fold_dataloaders = []
    for train_idx, test_idx in kfold.split(dataset):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)
        train_dataloader = DataLoader(train_subset, batch_size=batch_train, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=batch_test, shuffle=False)
        fold_dataloaders.append((train_dataloader, test_dataloader))
    return fold_dataloaders