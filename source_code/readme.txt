README


The files and folders for the project have to be in the following order to run correctly:

	Group_id_21/
		-report.pdf
		-source_code/		
			-data/
				-train-sample_October_9_2012_v2.csv [171.84 MB]
				-train_October_9_2012.csv           [  3.83 GB]				
			-Multiclass_Classifier_Vowpal_Wabbit/
				-csv2vw.ipynb
				-Sigmoid_mc_&_calculating_LogLoss_value.ipynb
			-Binary_Classifiers.ipynb
			-Cleaning_&_Preprocessing.ipynb
			-Exploring_&_Visualization.ipynb
			-Multiclass_Random_Forest_Classifier.ipynb
			-readme.txt

			
			
Implementation:


	DOWNLOADING THE DATA:
	
		The following data files have to be downloaded from the Kaggle website link below:
			Link: https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/data
			
			1.train-sample_October_9_2012_v2.csv [171.84 MB]
			2.train_October_9_2012.csv           [  3.83 GB]
			
	both the files have to be placed in the /Group_id_21/source_code/data/
	
	
	
	
	CLEANING AND PREPROCESSING:
	
		-open the jupyter notebook file /Group_id_21/source_code/Cleaning_&_Preprocessing.ipynb
		-all the lines in the notebook have to be executed twice, once each for 
			-large_dataset
			-small_dataset
		-which can be switching by altering the commented lines in the command line2 after importing the required libraries.
		
		
		FILES GENERATED:
			-Group_id_21/source_code/data/data_large.csv
			-Group_id_21/source_code/data/data_small.csv
			
	
		
	EXPLORING AND VISUALIZATION:
	
		-open the jupyter notebook file /Group_id_21/source_code/Exploring_&_Visualization.ipynb
		-on running the commands in jupyter notebook,
			-three scatter matrices:
				1. on features 'num_tags','body_len','owner_reputation','age'
				2. on features 'hw_in_title','hw_in_tags','hw_in_body'
				3. on features 'body_len','owner_reputation','age'
			-and a 3D scatter plot
				1. on features 'body_len','owner_reputation','age'
		are created.
		
	BINARY CLASSIFICATION:
	
		-open the jupyter notebook file /Group_id_21/source_code/Binary_Classifiers.ipynb
		-set the value for 'data_to_use', depending on data to be used for classification

			1: Balanced Dataset
			2: Unbalanced Dataset
			3: Undersampled Dataset

			eg:-  data_to_use = 1
			
		Then on executing the commands one by one , the following classifiers are run:
		
			1. Random Forest Classifier
			2. Naive Bayes Classifier
			3. Gradient Boosting Classifier
			4. Neural Network Classifier
			
		
		For each classifier, the follwing steps are executed:
			1. Training the model using train data
			2. Predicting the output using the model for test data
			3. Calculating the accuracy
			4. Computing the confusion matrix
			5. Computing the precision and recall
			6. Also outputs the time to train the data

			
	MULTICLASS CLASSIFICATION
	
		1. Random Forest Multiclass Classifier
			-open the jupyter notebook file /Group_id_21/source_code/Multiclass_Random_Forest_Classifier.ipynb
			-on executing the commands one by one , the following steps are are run:
				1. Partitioning of data into test and train
				2. Building the train classifier model using train data
				3. Predicting the output using the model for test data
				4. Calculating the accuracy
				5. Computing the confusion matrix
				6. Computing the precision and recall
				7. Also outputs the time to train the data
				8. Evaluates and outputs the logarithmic loss value over the predicated data
				
		2. Vowpal Wabbit
			
			Step 1. Converting the csv data to vowpal wabbit readable format
				-open the jupyter notebook file /Group_id_21/source_code/Multiclass_Classifier_Vowpal_Wabbit/csv2vw.ipynb
				-There are two sections in this jupyter notebook
					1. Creation of test and train data
						-this section needs to be run twice, once each for,
							-small data 
							-large data
							can be changed by adding commenting the and uncommenting the executable command 2
					2. Conversion of csv data to vowpal wabbit format
						- needs to be run 4 times, twice each for each data, small and large, and train and test for each data.
						- this creates the following four files:
							1. test_large.vw
							2. test_small.vw
							3. train_large.vw
							4. train_small.vw
							
			Step 2. Here we build a module using vowpal wabbit on train data as follows
			
					$ sudo apt-get install vowpal-wabbit
					
					$ cd /Group_id_21/source_code/Multiclass_Classifier_Vowpal_Wabbit/
					$ vw --loss_function logistic --oaa 5 -d train_large.vw -f model_large
					
			Step 3. Creating a raw prediction of the test data using the model created above
			
					$ vw --loss_function logistic -i model_large -t -d test_large.vw -r raw_predictions_large.txt
					
			Step 4. Normalizing the output generated above in raw_predictions_large.txt
			
					- open the jupyter notebook file Group_id_21/source_code/Multiclass_Classifier_Vowpal_Wabbit/Sigmoid_mc_&_calculating_LogLoss_value.ipynb
					- run the first half of the file that creates the normalized output
			
			Step 5. Calculating the MLL( logarithmic loss function )
			
					- continue running the jupyter notebook file run above (Group_id_21/source_code/Multiclass_Classifier_Vowpal_Wabbit/Sigmoid_mc_&_calculating_LogLoss_value.ipynb)
					- Log loss value is evaluated
			
					
							
			
			
				
		