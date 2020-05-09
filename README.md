# CS 5293, Spring 2020 Project 3

###### The Analyzer / Cuisine Predictor

###### Ana Marple

## Intro
-----------
This program allows the user to input a list of ingredients and recieve a cuisine prediction (such as 'mexican,' 'thai,' 'italian,' 'american,' etc.), followed by a recommendation list of 5 of the closest recipes to the ingredients provided. 

#### Installation
----------------
1. Install the package
```bash
pip install project3
```
2. Go into the shell
```bash
pipenv shell
```
3. Make sure neccessary packages are downloaded
```bash
python setup.py install
```
Note: Python 3.7 was used in the making of this program.

### Folder Structure
----------------------
Below is the tree structure for this project. The main module is predictor.py, which contains all of the neccessary functions. The data set was provided by Yummly.com and is stored as ```docs/yummly.json``` and was used to train the model predictor. The latest results are printed to the console, as well as written to ```docs/results.txt```.

```
.
├── COLLABORATORS
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── build
│   ├── bdist.linux-x86_64
│   └── lib
│       └── project3
│           ├── __init__.py
│           └── predictor.py
├── dist
│   └── project_3-1.0-py3.7.egg
├── docs
│   ├── requirements.txt
│   ├── results.txt
│   └── yummly.json
├── project3
│   ├── __init__.py
│   └── predictor.py
├── project_3.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   └── top_level.txt
├── setup.cfg
├── setup.py
└── tests
    ├── test_parse.py
    └── test_predict.py

```

#### Usage
------------
An example of the command used to execute the program can be seen below. Each ingredient can be added after entering the appropriate ```--ingredient``` tag. If an ingredient were to consist of two or more words, such as'hot dog,' make sure to wrap the word with quotation marks.

```bash
pipenv run python project3/predictor.py --ingredient tortillas --ingredient tacos --ingredient rice --ingredient 'guacamole'
```

#### Output
------------
The code above produces the output below. As you can see, the predicted cuisine was 'mexican' and then it lists 5 of the cloest recipes by their Id and then is followed by the distance in parenthesis.

```bash
Predicted Cuisine: mexican
Closest 5 recipes: 
1. Recipe Id: 4734 (0.35355339059327373)
2. Recipe Id: 31646 (0.3333333333333333)
3. Recipe Id: 44498 (0.31622776601683794)
4. Recipe Id: 26065 (0.2886751345948129)
5. Recipe Id: 14031 (0.2886751345948129)
```

## Code Flow
-------------------

### 1. Get input / Parse dataset
After the user inputs the ingredients to the program, the main function driver code stores the ingredients. It first calls the parse_data function to parse the yummly.json data file. The parse_data function takes in the name of the file to be parsed and loads the contents of the file using the json package. Each json object is then parsed and its respective id, cuisine, and ingredients are added to lists that are then used to create a pandas dataframe that is returned to the main code.

### 2. Convert the text to features / train and prepare classifiers
After getting the dataframe from the previous section, the program uses sklearn's train_test_split function to split the dataframe into separate testing and training sets since the size of the data is so large. Data was split into 25% train and 75% test, as proportions for splitting are recommended to be around 20/80 to 30/70 to prevent over/underfitting the data.

After splitting the data, the resulting train and test ingredients and labels were input to the get_model function. The get_model function first vectorizes the ingredients data from the training set using sklearn's CountVectorizer so that the text data is converted into features to create a prediction model. Ultimately, support vector machines were used to create the cuisine classification model, as they usually have higher accuracy scores (at least according to the classification model results in ***Text Analytics with Python***, pg. 317). Although, SVMs are usually slower, so the part of the code where the accuracy and test accuracy scores were calculated are commented out. It was found that after testing the model on the test df that it had an accuracy of 73%, which is relatively high. The get_model function returns the countVectorizer and svm model to the main code.

### 3. Prediction
The main code directly takes the countVectorizer, svm model, and user input ingredients into the predict function. There, it uses the countVectorizer to vectorize/featurize the input ingredients so that the svm model can predict the cuisine for them. The ingredients and predicted cuisine are stored in a single dataframe row with an Id of 'User_input.' Reasons for this will be explained in the next section. The function then prints and returns the predicted cuisine and the row of data.

### 4. Select closest N recipes / Give output
Before computing the distance of recipes from each other, the train dataframe of around 9700 recipes ***was reduced to 5000, so that sklearn's cosine_similarity could run*** in the get_recipes function (otherwise there was not enough memory and the program would return 'Killed'). The 'User_input' dataframe row was appended to the further reduced train dataframe and a new CountVectorizer was fit for that dataframe to compute its features. The get_recipes function was then called and took in the new train dataframe, its features, the user input ingredients, and the predicted cuisine. 

The get_recipes function first used the features to prepare a cosine similarity matrix and then turn it into a dataframe. The index of 'User_input' was found to get the row of cosine similarities between it and all the other recipes in the cosine similarity matrix. The row of distances was then sorted and the top 5 scores (besides the score with itself) and the index of those recipes were stored to retrieve the recipe Ids. 5 was chosen as the N since that was what was in the project instructions example and that 5 seems like a safe/reasonable number of recommendations (don't want to overwhelm the user with too many recipe recommendations). This information was then printed to the console and also written to the docs/results.txt. 

## Testing
------------
There are four tests to diagnose the program, all located in the ```tests/``` folder. There is a 'test_parse.py' to test that the yummly.json file was properly parsed into a dataframe. The 'test_predict.py' contains two tests that test the remaining three functions. The test_predict function uses a predetermined list of ingredients to make sure that 1) the get_model function returns the count vectorizer and svm model to 2) input into the predict function and return a predicted cuisine value. The other test_recipes test function makes sure that the 5 recommended recipes and their distances are printed out to results.txt.

Command to run pytest:
```bash pytest -p no:warnings -s```


#### References
------
***Text Analytics with Python***, Second Edition, by Dipanjan Sarkar
https://machinelearningmastery.com/make-predictions-scikit-learn/
