
import numpy as np
import nibabel as nib
import os
import inference
from tabulate import tabulate
import pandas as pd


'''
Script for independent evaluation of segmented samples (predicted vs ground truth)
Please specify paths to folders with predicted and ground truth folders directly in this script.
The file names in both folders must be identical.
Also images must have identical width and height (as looking into crossections perpendicular to grain length)
'''

def load_set(pred_path, prediction_filenames):
    pred_list = []
    for pfname in prediction_filenames:
        source_pt = os.path.join(pred_path, pfname)
        img_nii = nib.load(source_pt)
        img_npt = img_nii.get_fdata(dtype=np.float32).astype(np.uint8)        
        img_npt = np.squeeze(img_npt)        
        pred_list.append(img_npt)
        
    return pred_list


def main():
    folder_gt = "../data/demo_data_for_2D_models/test/label"    
    folder_predictions = "../data/demo_data_for_2D_models/prediction_nnunet"
    save_file = "evaluation.xlsx"
    file_names = sorted(os.listdir(folder_gt))
    print(f"File names: {file_names}") # must match those in the prediction folder

    gt_list = load_set(folder_gt, file_names)
    gt_total = np.concatenate(gt_list)
    print(f"{len(gt_list)} = ")
    print(f"{gt_total.shape = }, {gt_total.dtype = }")

    predictions_list = load_set(folder_predictions, file_names)
    print(f"{len(predictions_list) = }")
    predictions_total = np.concatenate(predictions_list)
    print(f"{predictions_total.shape = }, {predictions_total.dtype = }")

    
    iou = inference.iou_score_total(gt_total, predictions_total, with_pericarp = False)
    dice = inference.dice_score_total(gt_total, predictions_total, with_pericarp = False)
    metrics_total = iou.join(dice)
    metrics_total['sample'] = 'total'
    metrics_total['class'] = metrics_total.index
    metrics_total = metrics_total.reset_index()
    metrics_total = metrics_total[['sample', 'class', 'IoU', 'Dice']]
    print("\n" + tabulate(metrics_total, headers='keys', tablefmt='psql', showindex=True))     
    
    dfs = [metrics_total]
    
    for i, file in enumerate(file_names):
        print(file)
        iou = inference.iou_score_total(gt_list[i], predictions_list[i], with_pericarp = False)
        dice = inference.dice_score_total(gt_list[i], predictions_list[i], with_pericarp = False)
        metrics = iou.join(dice)
        metrics['sample'] = file
        metrics['class'] = metrics.index
        metrics = metrics.reset_index()
        metrics = metrics[['sample', 'class', 'IoU', 'Dice']]
        dfs.append(metrics)
        print("\n" + tabulate(metrics, headers='keys', tablefmt='psql', showindex=True))   

    dfs = pd.concat(dfs, axis = 0,  ignore_index = True)
    #print("\n" + tabulate(dfs, headers='keys', tablefmt='psql', showindex=True))
    dfs.to_excel(save_file)
    
if __name__ == '__main__': 
    main()


 


