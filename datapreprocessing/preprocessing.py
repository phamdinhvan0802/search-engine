import json
import nltk
import math
import string
from nltk.corpus import stopwords
from nltk.stem.porter import * 
import pickle # save and load data
from collections import OrderedDict # dict library

def read_and_preprocessing(json_filename, num_attribute, is_lower_case, is_stem,is_remove_stopwords, is_remove_puctuation, stemmer):
    '''
    inputs: jason_filename: data
    num_attribute: set to 11 at the end to include recipe w/ 11 attributes (title, ingredients, etc.)
    is_lower_case: flag, logical
    is_stem,is_remove_stopwords: flag, logical
    is_remove_puctuation: flag, logical
    stemmer: an object for stemming, returned from PorterStemmer, which is a function in nltk.stem.Porter
    '''
    data = json.load(open(json_filename)) # a list of recipes
    documents = [] # initialize a list (of list), each entry is one recipe's ingredients with stemmed key words
    index_in_json = [] # record the index of selected recipes in json file
    title_set = set() # a set of recipe titles (string), in case there are duplicated recipes in json file
    word_set = set() # a set of word of ingredients
    cnt = 0 # count
    len_data = str(len(data)) # number of recipes in json file

    for i in range(0, len(data)): # iterate through recipes
        if len(data[i]) == num_attribute and len(data[i]['ingredients']) != 0: # select recipes that have 11 attibutes and non-zero ingredients
            if data[i]['title'] not in title_set: # there are repeated recipes
                # print(str(i) + '/' + len_data)
                title_set.add(data[i]['title']) # add current recipe title into title_set
                index_in_json.append(i) # record index of current recipe, a list
                ingredients = data[i]['ingredients'] # get ingredient (a list of strings)

                # data preprocessing starts:
                actual_ingredients = [] # list of preprocessed ingredient words
                for each_ingredient in ingredients: # iterate through ingredients (a list of strings), each ingredients is one string
                    if is_lower_case:
                        each_ingredient = each_ingredient.lower()

                    tokens = nltk.word_tokenize(each_ingredient) # tokenize each string into words

                    if is_stem:
                        singles = [stemmer.stem(token) for token in tokens] # stemming token

                    if is_remove_stopwords:
                        filtered_words = [word for word in singles if word not in stopwords.words('english')]
                    else:
                        filtered_words = singles

                    filtered_words_2 = []
                    if is_remove_puctuation:
                        for word in filtered_words:
                            if word.isalpha(): # if alphate=not puctuation
                                filtered_words_2.append(word)
                                word_set.add(word) # save the preprocced word
                    else:
                        filtered_words_2 = filtered_words
                    # print('-----------------------------------')
                    # print(filtered_words_2)
                    actual_ingredients = actual_ingredients + filtered_words_2
                documents.append(actual_ingredients)



    return index_in_json, documents, word_set



def save_func(filename, data): # save the generated files into the same dir
    # print('start saving ' + filename)
    with open(filename, "wb") as fp1:
        pickle.dump(data, fp1)
    # print('end saving ' + filename)

def load_func(filename): # load the saved files
    # print('start loading ' + filename)
    with open(filename, "rb") as fp1:
        data = pickle.load(fp1)
    # print('end loading ' + filename)
    return data


stemmer = PorterStemmer() # PorterStemmer is a function in nltk.stem.Porter that returns an object for stemming
is_lower_case = True
is_stem = True
is_remove_stopwords = True
is_remove_puctuation = True

name_documents = 'documents'
name_index_in_json = 'index_in_json'
name_word_set = 'word_set'
name_inverted_index = 'inverted_index'
num_attribute = 11 # only include recipe w/ 11 attributes (title, ingredients, etc.)
json_filename = 'full_format_recipes.json'
name_doc_length = 'doc_length'


# index_in_json, documents, word_set = read_and_preprocessing(json_filename, num_attribute, is_lower_case, is_stem,is_remove_stopwords, is_remove_puctuation, stemmer)
# save_func(name_documents, documents)
# save_func(name_index_in_json, index_in_json)
# save_func(name_word_set, word_set)

load_documents = load_func(name_documents)
load_index_in_json = load_func(name_index_in_json)
load_word_set = load_func(name_word_set)

print(load_documents)
print(load_index_in_json)
print(load_word_set)


# next: 1.remove costomized stop words
# 2. construct inverted index generated base on documents (dict with each key being actual_ingredients): {egg: [（index1, num1）, (index2,num2)...], tomato....
# 3. vector space model
