# Evaluation

With the *evaluation.py* script, you can compare two sets of data - predicted segmentation and ground truth, and calculate total *Intersection over Union* and *Dice* scores, as well as sample-wise reports. The folders are specified directly in the script, and the names of samples should match to correctly identify corresponding pairs. 
Here are placed *Excel* files with the scores for the 2D model, 2.5D(stacked) model, and *nnUnet*, trained on the same amount of data: 18 images that were available in the middle of the project - the same images are packed as demo data for downloading (Please note that *nnUnet* checkpoints offered for downloading are from the latest iteration of training, so you will probably get slightly different results, but with respect to wheat data,  represented in the test set of demo data, the performance is practically the same).


