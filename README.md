# Deep_Transfer

## Introduction
    source code for paper https://link.springer.com/article/10.1007/s00521-018-3468-3

    Deep transfer learning for military object recognition under small training set condition

    training data in image200d, testing data in sample_images

    output_graph_new.pb is model file which is get only by training model with fully conneted layer 200 steps.
    output_labels_new.txt record the labels in our paper
 
## Train:
    our code is default for training fully conected layer, if you want train more layer, you can replace retrain.py by that in other layer,and then 
    
    CUDA_VISIBLE_DEVICES=0 python retrain.py
    
## Test:
    we make a sample visual interface for testing
    
    you can only 
    
    python ./open.py
    
    and you can see 
    ![image](Deep_Transfer/ShowImage/image1.png)
    
    then click open image, chose an image for test
    ![image](Deep_Transfer/ShowImage/image1.png)
    
    last,click test,you can get predict result
    ![image](Deep_Transfer/ShowImage/image1.png)
    
    
    
