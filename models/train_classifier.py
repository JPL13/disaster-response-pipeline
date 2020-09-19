import sys
import os
import re
import pandas as pd
from sqlalchemy import create_engine
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier



def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    
    path, file = os.path.split(database_filepath)
    
    table_name = file.replace(".db","") + "Table"

    df = pd.read_sql_table(table_name, engine)
    X = df.iloc[:, 1]
    y = df.iloc[:, 4:]
    
    return X, y, y.columns 



def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok not in stop_words]


    return clean_tokens
    


def build_model():
    svc_pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier( OneVsRestClassifier(LinearSVC(), n_jobs=1)))
            ])

    parameters = {
        'clf__estimator__estimator__loss': ('hinge', 'squared_hinge'),
        'clf__estimator__estimator__C': (0.5, 1.0)
    } 


    cv = GridSearchCV(estimator=svc_pipeline, n_jobs = -1, param_grid=parameters)
        
    model = cv


    return model




def evaluate_model(model, X_test, Y_test, category_names):
    y_predicted = model.predict(X_test)

    y_predicted = pd.DataFrame(y_predicted)

    y_predicted.columns = Y_test.columns
    y_predicted.index = Y_test.index

    # overall accuracy
    accuracy = (y_predicted == Y_test).mean().mean()
    print('Overall Accuracy {0:.2f}% \n'.format(accuracy*100))

    # Classification report
    #print(classification_report(Y_test, y_predicted, target_names=category_names))


    for i in range(36):
        print('Classification report for "{}" \n'.format(y_predicted.columns[i]))
        print(classification_report(Y_test.iloc[:, i], y_predicted.iloc[:, i]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()