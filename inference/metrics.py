import numpy as np
import pandas as pd

def iou_score_total (y_true, y_pred, with_pericarp = True):
    assert np.array_equal(np.unique(y_pred), np.array([0, 1, 2, 3, 4])), f"Incorrect prediction values, expected [0, 1, 2, 3, 4], got {np.unique(y_pred) = }"
    assert np.array_equal(np.unique(y_true), np.array([0, 1, 2, 3, 4])), f"Incorrect label values, expected [0, 1, 2, 3, 4] got {np.unique(y_true) = }"
    
    titles = ['1 embryo', '2 endosperm', '3 aleuron']
    if with_pericarp:
        titles = titles + ['4 pericarp']
    else:
        y_pred = np.where(y_pred == 4, 0, y_pred)
        y_true = np.where(y_true == 4, 0, y_true)        
    num_cl = np.unique(y_pred)[1:]
    #score = np.zeros(num_cl.shape[0]+1)
    score = []
    for cl in num_cl:
        y_true_cl = np.where(y_true==cl, 1, 0)
        pred_cl= np.where(y_pred==cl, 1, 0) 
        intersection = np.logical_and(y_true_cl, pred_cl)
        union = np.logical_or(y_true_cl, pred_cl)
        score.append(round(np.sum(intersection) / np.sum(union),4))
        #print(f"IoU score for class{cl} = {round(score[i], 4)}") 
    #print(f"Total IoU score: {round(np.mean(score), 4)}")
    score.append(round(np.mean(score), 4))    
    titles = titles + ["mean"]
    data = {"IoU":score}
    df = pd.DataFrame(data, index = titles)
    return df

def dice_score_total (y_true, y_pred, with_pericarp = True):
    assert np.array_equal(np.unique(y_pred), np.array([0, 1, 2, 3, 4])), "Incorrect prediction values, expected [0, 1, 2, 3, 4]"
    assert np.array_equal(np.unique(y_true), np.array([0, 1, 2, 3, 4])), "Incorrect label values, expected [0, 1, 2, 3, 4]" 

    titles = ['1 embryo', '2 endosperm', '3 aleuron']
    if with_pericarp:
        titles = titles + ['4 pericarp']
    else:
        y_pred = np.where(y_pred == 4, 0, y_pred)
        y_true = np.where(y_true == 4, 0, y_true)     

    num_cl = np.unique(y_pred)[1:]
    score = []
    for cl in num_cl:
        y_true_cl = np.where(y_true==cl, 1, 0)
        pred_cl= np.where(y_pred==cl, 1, 0) 
        intersection = np.logical_and(y_true_cl, pred_cl)
        union = np.logical_or(y_true_cl, pred_cl)
        score.append(round(2 * np.sum(intersection) / (np.sum(y_true_cl) + np.sum(pred_cl)), 4))        
    score.append(round(np.mean(score), 4))  
    titles = titles + ["mean"]
    data = {"Dice":score}
    df = pd.DataFrame(data, index = titles)
    return df