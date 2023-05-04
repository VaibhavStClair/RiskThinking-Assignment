# What is this project about?

This project involves creating a data pipeline that is used for deploying machine learning model.
It uses Spark and Python for data integration and transformations. 

The Project is divided into 4 phases:


>Raw Data Processing

>Feature Engineering

>Integrate ML training

>Model Serving

For phase 1 and 2, PySpark was used to effectively Extract, Transform and Load the data.

For phase 3 and 4, Python and Flask API was used to fit and deploy the model.

# Steps of Execution

This section explains the extract transform load for the part one and two of the project. 
First we'll start by creating the spark session. 
So we have to load the data inside spark for parallel computation and processing, for that we are first defining the schema for both the datasets. 
Next step is to declare the path where the data is residing and where it should be stored after transformations. 

There are 3 main functions have built in Utilities/ETL.py:
1. load_files: Here, a dictionary for which the key will be the name of the file and the value will be the data frame. 
I tried many methods for loading the data sets into memory like repartition or coalesce or changing the config parameters while loading, but they all seem to give me out of memory issue. 
So for now I have taken first 40 files in both the datasets due to my laptop's limited RAM and cores. 

NOTE: ***If you want to run all the files, please comment line 90,93, 102 and 103 in ETL.py file***

2. writeparquet: A dictionary with key and value pair, writing the data frames as the value with the name as keys of the dictionary into storage. 

3. transformation: this will take the date column in string format and convert it to timestamp first and then converting it back to the date format. 
After that I am getting the month and year from this date column. We need to define the windows function here so that we can partition the data for calculating the moving average and the ruling median. 
For that the partition is being done by month year and symbol will stop the moving average and the ruling medium are calculated as required. 
All these functions are in a class and to call these I'm just making an object and calling the class. 
The next step is to store the final data which is union of both the stocks and the ETF's data set, and I'm storing it into a different directory. 

The files that are downloaded from the kaggle are being stored in the bronze directory, after transformation and applying rolling median and moving average the data are stored in silver directory, and finally the union of the data sets are stored in the gold directory. 
The next step is to apply the machine learning models as given by the document for that we are importing the final data set from the gold layer and then applying the random forest regressor. 
Here the data was split in training and testing for 80% in training and 20% in testing. 
After fitting the model to the training data set the prediction variable is made on the test data set. 
The machine learning model outputs the prediction variable, the mean absolute error, and the mean squared error. 
I am also making a directory to store these results and the model. 
For deployment purpose, I am taking three variables as part of the deployment with the first one being the rolling median the second one being the moving average and the third one being the predicted value. 

Flask integration is used for the deployment, in which index.html is for welcome page and result.html is for showcasing the results. 
User will give the average and the rolling median as input and it ll generate the volume predictions.

Screenshots,
Before ETL:

![/assets/images/img_5.png](/assets/images/img_5.png)

After running ETL.py:
![/assets/images/img_5.png](/assets/images/img_3.png)
![/assets/images/img_5.png](/assets/images/img_4.png)

After running main.py
![/assets/images/img_5.png](/assets/images/img_6.png)

![/assets/images/img_5.png](/assets/images/img_7.png)

Log files generated after Part 4:
![/assets/images/img_5.png](/assets/images/img_8.png)

# How to run on your machine

1. Clone the project and place your ETF and Stocks data files in '../VolumePrediction/bronze/' (for e.g ../VolumePrediction/bronze/etf)
2. Run pip install -r requirements.txt for installing libraries
3. For Part 1 and 2, execute ETL.py inside Utilities folder ***without this, step 4 would not execute***
4. For Part 3 and 4, execute main.py file, this will generate localhost URL, please click it and start using the website
5. After entering the dummy values for Average and Rolling median, click submit to get the predicted Volume
6. Verify log files, Silver, and Gold folders created under '../VolumePrediction/' directory



References:
https://stackoverflow.com/questions/32336915/pyspark-java-lang-outofmemoryerror-java-heap-space
https://sparkbyexamples.com/pyspark/pyspark-read-and-write-parquet-file/
https://gist.github.com/malanb5/0179157f732c9765c3bf6023ebbf2077
https://sparkbyexamples.com/pyspark/pyspark-window-functions/
https://www.learneasysteps.com/how-to-calculate-median-value-by-group-in-pyspark/
https://www.linkedin.com/pulse/time-series-moving-average-apache-pyspark-laurent-weichberger/
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
https://www.geeksforgeeks.org/deploy-machine-learning-model-using-flask/
