from spectrum import Spectrum
import util
from tqdm import tqdm

class SweepUp:
    
    @staticmethod
    def get_all_images():
        all_inputs, all_masks, all_preds = [], [], []

        total_iterations = len(util.MODELS) * len(util.FOLDS)

        with tqdm(total=total_iterations, desc="Reading Images") as pbar:
            for model in util.MODELS:
                for fold in util.FOLDS:
                    inputs, masks, preds = Spectrum.inputs_masks_preds(model=model, fold=fold)
                    all_inputs.extend(inputs)
                    all_masks.extend(masks)
                    all_preds.extend(preds)

                    pbar.update(1)
       
        return all_inputs, all_masks, all_preds
    
