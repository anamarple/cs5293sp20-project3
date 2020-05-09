from project3 import predictor


#Tests the parse_data function to make sure .json file data is read and parsed correctly
def test_parse():
    
    file_name = 'docs/yummly.json'
    
    #Parse .json file into df
    df = predictor.parse_data(file_name)
    print()
    print(df)

    #Assert that df is not empty
    assert(len(df) > 0)
    

