from parser_file_parameter import Parser, File, Parameter
from load_dataset import LoadDataset
from select_set import SelectSet
from select_model import SelectModel
from optimizer_scheduler import OptimizerScheduler
from epoch import Epoch
from metric import Metric
from print import Print
import numpy as np
import util
import torch
                        
class Start: 
    @staticmethod
    def run():
        parser = Parser.init()
        _sweep_up(args=parser.args)
                         
def _sweep_up(args):
    for spec_name in util.SPECTRUMS:
            for mod_name in util.MODELS:
                    
                file = File.build(args=args, model_name=mod_name, spectrum_name=spec_name)  
                param = Parameter(model_name=mod_name, spectrum_name=spec_name, 
                                  save_name=file.save_name)
                load_dat = LoadDataset.from_files()
                select_set = SelectSet.separate_them(load_dat=load_dat)
                _same_random_numbers()
                select_model = SelectModel.choose_one(model_name=mod_name, 
                                                      spectrum_name=spec_name)
                opt_sched = OptimizerScheduler.load(args=args, model=select_model.model)
                
                Epoch.run(args=args, file=file, spectrum_name=param.spectrum_name, 
                            sel_set=select_set, model=select_model.model, 
                            opt_sched=opt_sched)
                
                metr = Metric.evaluate(args=args, file=file, spec_name=param.spectrum_name, 
                                       dataloader=select_set, model=select_model.model)
                
                Print.screen_and_file(param=param, metr=metr) 
                        
def _same_random_numbers():
    torch.manual_seed(0)
    np.random.seed(0)   
    
if __name__ == "__main__":
    Start.run()              