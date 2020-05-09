import argparse
import pandas as pd
import numpy as np


###################################################################
from sklearn.model_selection import train_test_split

#Main driver that takes in the list of ingredients from user input and calls the other functions
def main(ingredients):
    
    #print(ingredients)        
    #Reads .json file and returns dataframe of id, cuisine, and ingredients
    df = parse_data('docs/yummly.json')

    #Split df into train (1/4) and test (3/4) data
    test_ing, train_ing, test_label, train_label, test_id, train_id = train_test_split(np.array(df['Ingredients']), np.array(df['Cuisine']), np.array(df['Id']), test_size = 0.25, random_state = 42)
     
    #Get vecorizer and model to predict cuisine
    cv, svm = get_model(train_ing, test_ing, train_label)

    #Predict cuisine from user input
    cuisine, input_row = predict(ingredients, cv, svm)

    #Select closest N recipes - prints results to terminal and file
    #***Had to shorten recipe list to 5000 otherwise cosine_similarity wouldn't run
    data = {'Id' : train_id[:5000], 'Cuisine' : train_label[:5000], 'Ingredients' : train_ing[:5000]}
    tr_df = pd.DataFrame(data)
     
    #Add user input to train data
    tr_df = tr_df.append(input_row, ignore_index = True)
    tr_features = cv.fit_transform(tr_df['Ingredients'])
    get_recipes(tr_df, tr_features, ingredients, cuisine)


###################################################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

#eturns features and svm model used to predict the cuisine
def get_model(train_ing, test_ing, train_label):

    #Get features                                           
    cv = CountVectorizer(binary = False, min_df = 0.0, max_df = 1.0)
    train_features = cv.fit_transform(train_ing)
    test_features = cv.transform(test_ing)
    
    #Train / Prepare Classifiers
    #SVM:
    svm = LinearSVC(penalty = 'l2', C =1, random_state = 42, max_iter = 10000)
    svm.fit(train_features, train_label)
        
    '''
    #***Commented out to save compile time
    svm_bow_cv_scores = cross_val_score(svm, train_features, train_label, cv = 5)
    svm_bow_cv_mean_scores = np.mean(svm_bow_cv_scores)
    
    print('CV Accuracy:', svm_bow_cv_scores)
    print('Mean CV Accuracy:', svm_bow_cv_mean_scores)
    
    svm_bow_test_score = svm.score(test_features, test_label)
    print('Test Accuracy:', svm_bow_test_score)  #0.73
    y = svm.predict(test_features)
    results = {'Ingredients' : test_ing, 'Cuisine' : test_label, 'Predicted' : y}
    results_df = pd.DataFrame(results)
    print(results_df)
    '''
    
    #Return vectorizer and model
    return(cv, svm)


###################################################################
#Takes ingredients, vectorizer, and model and returns predicted cuisine
def predict(ingredients, vec, model):

    ingredients_ = []
    ingredients_.append(' '.join(ingredients))
    input_features = vec.transform(ingredients_)
    
    #Predict cuisine
    cuisine = model.predict(input_features)
  
    #Create new data row
    input_row = {'Id' : 'User_input', 'Cuisine' : cuisine[0], 'Ingredients' : ingredients_[0]}

    print('Predicted Cuisine: ' + cuisine[0])
    return(cuisine, input_row)


###################################################################
from sklearn.metrics.pairwise import cosine_similarity

#Prints out 5 closest recipes to ingredients (Id and cosine similarity distance)
def get_recipes(df, matrix, ingredients, cuisine):

    doc_sim = cosine_similarity(matrix)
    doc_sim_df = pd.DataFrame(doc_sim)
    #print(doc_sim_df.head())
    
    #Get index of 'User_input'
    recipes_list = df['Id'].values
    recipe_idx = np.where(recipes_list == 'User_input')[0][0]

    #Get list of cosine similarities for user input/sort and store the cosine similarity score
    recipe_similarities = doc_sim_df.iloc[recipe_idx].values
    
    #Create / open file to write results to
    f = open('docs/results.txt', 'w+')
    f.write('\nInput Ingredients: ' + str(ingredients) + '\n')
    f.write('Predicted Cuisine: ' + str(cuisine[0]) + '\n')
    f.write('Closest 5 recipes: \n')

    #Print top 5 recipes and cosine similarity distance
    similar_recipe_idxs = np.argsort(-recipe_similarities)[1:6]
    print('Closest 5 recipes: ')
    i = 1
    for idx in similar_recipe_idxs:
        f.write(str(i) + '. Recipe Id: ' + str(recipes_list[idx]) + ' (' + str(doc_sim_df[idx][5000]) + ') \n')       
        print(str(i) + '. Recipe Id: ' + str(recipes_list[idx]) + ' (' + str(doc_sim_df[idx][5000]) + ')')
        i = i + 1
    
    f.close()


###################################################################
import json

#Reads yummly.json file and parses it into a dataframe object
def parse_data(file_):
    
    ID = []
    cuisine = []
    ingredients = []

    #Loads and reads .json file
    f = open(file_, 'rb')
    content = json.loads(f.read())
    for obj in content:
        ID.append(obj['id'])
        cuisine.append(obj['cuisine'])
        ingredients.append(' '.join(obj['ingredients']))

    f.close()

    #Builds dataframe
    data = {'Id' : ID, 'Cuisine' : cuisine, 'Ingredients' : ingredients}
    df = pd.DataFrame(data)
    
    #print(df)
    return(df)
    

###################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Supply list of ingredients and receive predicted cuisine type')
    parser.add_argument('--ingredient', action = 'append', help = 'Enter each ingrdient available')

    args = parser.parse_args()
    if args.ingredient:
        main(args.ingredient)
