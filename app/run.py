import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    category_counts = category_counts.sort_values(ascending = True)
    category_names = list(category_counts.index)
    
    direct_counts=[]
    news_counts=[]
    social_counts=[]
    
    for colname in category_names:
        count1=df[df['genre']=='direct'][colname].sum()
        count2=df[df['genre']=='news'][colname].sum()
        count3=df[df['genre']=='social'][colname].sum()
    
        direct_counts.append(count1)
        news_counts.append(count2)
        social_counts.append(count3)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            "data": [
              {
                "type": "bar",
                "x": direct_counts,
                "y": category_names,
                "orientation": 'h', 
                "name" : 'direct',  
                 "marker": {
                        "color": 'rgba(55,128,191,0.6)',
                        "width": 1
                  }, 
                  
              },
                
             {
                "type": "bar",
                "x": news_counts,
                "y": category_names,
                "name" : 'news',   
                "orientation": 'h', 
                 "marker": {
                        "color": 'rgba(255,153,51,0.6)',
                        "width": 1
                  }, 
                  
              },
                
            {
                "type": "bar",
                "x": social_counts,
                "y": category_names,
                "name" : 'social',   
                "orientation": 'h', 
                 "marker": {
                        "color": 'rgba(50,171,96,0.6)',
                        "width": 1
                  }, 
                  
              }        
                
            ],
            "layout": {
              "title": "Messages by Category",
              
              'yaxis': {
                  
              },
              'xaxis': {
                  'title': "Count",
                  
              },
               "barmode": 'stack',
                
               "margin": {
                     "l": 200,
                     "r": 20,
                     "t": 50,
                     "b": 70
                    },
                "width": 1000,
                "height": 1000,   
              
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()