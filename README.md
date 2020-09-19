# Disaster Response Pipeline Project

This project analyzes disaster data from [Figure Eight](https://appen.com) to build a model for an API that classifies disaster messages.

The data set used in this project contains real messages that were sent during disaster events. This project builds a machine learning pipeline to categorize these events so that the messages could be send to an appropriate disaster relief agency.


### Project components:

#### 1. ETL Pipeline
This is a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### 2. ML Pipeline
This is a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### 3. Flask Web App
This project includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app will also display visualizations of the data.

Below are a few screenshots of the web app.

(https://github.com/JPL13/disaster-response-pipeline/blob/master/image/Overview.png)
(https://github.com/JPL13/disaster-response-pipeline/blob/master/image/category.png)
(https://github.com/JPL13/disaster-response-pipeline/blob/master/image/search-bar.png)
(https://github.com/JPL13/disaster-response-pipeline/blob/master/image/search-result.png)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
