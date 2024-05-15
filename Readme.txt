CWTImages and CWTImagesTest contains pre-generated images of seq len 120 for all datasets
Results fold contains results from different combinations of tests run
Data contains the raw measurment data sorted and saved as dataframes


To generate the Scalograms CWT_Gen_imgSubPlot.py is used to create the images with 6 subplots (one for each measurement)

To classify using CWT CWT_Training_General.ipynb 
To cluster using CWT, first cwt_cluster_modified.py is run to create the model with prediction. THen CWT_Gen_imgSubPlot.py is run to create the bar charts
To classify using LSTM, the LSTM General.ipynb is run
To cluster using DTW, the DTW_Clustering.ipynb is run