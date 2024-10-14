from dataclasses import dataclass
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import util
import threading

@dataclass
class Image:
    name: str
    model: str
    spectrum: str
    value: np.ndarray
    fold: str
     
    @classmethod
    def from_file(clc, name, model, spectrum=util.DEFAULT_SPEC, fold=util.DEFAULT_FOLD):
        file_paths = _build_path(model=model.lower(), spectrum=spectrum.upper(), fold=fold)  
        values = _get_values_with_threads(file_paths=file_paths, name=name)
    
        return clc(name=name, model=model, spectrum=spectrum, value=values, fold=fold)
         
def _build_path(model, spectrum, fold):
    parent = Path(util.DIR.PARENT)
    child = Path(f"{model.lower()}_{spectrum.upper()}_fold_{fold}")
    grandson_train = Path(util.DIR.GRANDSON_TRAIN)
    grandson_test = Path(util.DIR.GRANDSON_TEST)

    
    file_path_train = parent / child / grandson_train 
    file_path_test = parent / child / grandson_test 
    
    assert file_path_train.exists(), "Path does not exist"   
    assert file_path_test.exists(), "Path does not exist"   
    return file_path_train, file_path_test

def _get_values_with_threads(file_paths, name):
    file_path_train, file_path_test = file_paths

    # Usar multithreading para paralelizar a leitura de arquivos
    with ThreadPoolExecutor() as executor:
        # Submeter as tarefas de leitura de arquivos para o conjunto de treino
        futures_train = [executor.submit(_read_npz, file, name) for file in file_path_train.glob(util.DIR.EXTENSION)]
        
        # Submeter as tarefas de leitura de arquivos para o conjunto de teste
        futures_test = [executor.submit(_read_npz, file, name) for file in file_path_test.glob(util.DIR.EXTENSION)]
        
        # Coletar os resultados
        value_train = [future.result() for future in futures_train]
        value_test = [future.result() for future in futures_test]
    
    # Converter para arrays numpy
    value_train = np.array(value_train)  
    value_test = np.array(value_test)
    
    return value_train, value_test
                       
def _read_npz(file, name):   
    with np.load(file) as npz_file:
        return npz_file[name.lower()]


def _get_values(file_paths, name):
    value_train, value_test = [], []
    file_path_train, file_path_test = file_paths
    
    for file in file_path_train.glob(util.DIR.EXTENSION):
        samples = _read_npz(file, name) 
        value_train.append(samples)

    for file in file_path_test.glob(util.DIR.EXTENSION):
        samples = _read_npz(file, name) 
        value_test.append(samples)

    value_train = np.array(value_train)  
    value_test = np.array(value_test)
    
    return value_train, value_test