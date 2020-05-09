from project3 import predictor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#Tests the get_model and predict functions
def test_predict():

    file_name = 'docs/yummly.json'

    #Parse .json file into df
    df = predictor.parse_data(file_name)

    #Split df into test and train dataset        
    test_ing, train_ing, test_label, train_label, test_id, train_id = train_test_split(np.array(df['Ingredients']), np.array(df['Cuisine']), np.array(df['Id']), test_size = 0.25, random_state = 42)

    #Get vectorizer and model
    cv, svm = predictor.get_model(train_ing, test_ing, train_label)
    ingredients = ['taco', 'tortillas', 'guacamole', 'rice']

    print()
    #Predict cuisine from user input
    cuisine, input_row = predictor.predict(ingredients, cv, svm)
    
    #Assert that cuisine is not empty
    assert(len(cuisine) > 0)



import os
#Tests the get_recipes function
def test_recipes():

    file_name = 'docs/yummly.json'

    #Parse .json file into df
    df = predictor.parse_data(file_name)

    #Split df into test and train dataset        
    test_ing, train_ing, test_label, train_label, test_id, train_id = train_test_split(np.array(df['Ingredients']), np.array(df['Cuisine']), np.array(df['Id']), test_size = 0.25, random_state = 42)

    #Get vectorizer and model
    cv, svm = predictor.get_model(train_ing, test_ing, train_label)
    ingredients = ['taco', 'tortillas', 'guacamole', 'rice']

    print()
    #Predict cuisine from user input
    cuisine, input_row = predictor.predict(ingredients, cv, svm)

    #Select closest N recipes - prints results to terminal and file
    #***Had to shorten recipe list to 5000 otherwise cosine_similarity wouldn't run
    data = {'Id' : train_id[:5000], 'Cuisine' : train_label[:5000], 'Ingredients' : train_ing[:5000]}
    tr_df = pd.DataFrame(data)

    #Add user input to train data
    tr_df = tr_df.append(input_row, ignore_index = True)
    tr_features = cv.fit_transform(tr_df['Ingredients'])

    #Call get_recipes function
    predictor.get_recipes(tr_df, tr_features, ingredients, cuisine)


    #Assert thta results.txt file was written to
    assert(os.path.exists('docs/results.txt'))
