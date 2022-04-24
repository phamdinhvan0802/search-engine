## Original, model learned from https://en.wikipedia.org/wiki/Okapi_BM25.
## This file contains code for constructing the BM25 model for document 
## ranking and returns ranked recipes for a given query.


import json
import nltk
import matplotlib.pyplot as plt
import math
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *
import csv
import pickle
from sklearn import preprocessing
import operator
from collections import OrderedDict


def read_and_preprocessing(json_filename, num_attribute, is_lower_case,
    is_stem,is_remove_stopwords, is_remove_puctuation,
    stemmer, customized_stopwords):
    '''
    This function is used to preprocessing json data and only 
    remain ingredients
    inputs:
        jason_filename: data
        num_attribute: set to 11 at the end to include recipe
            w/ 11 attributes (title, ingredients, etc.)
        is_lower_case: flag, logical
        is_stem,is_remove_stopwords: flag, logical
        is_remove_puctuation: flag, logical
        stemmer: an object for stemming, returned from PorterStemmer,
            which is a function in nltk.stem.Porter
        customized_stopwords: special stopwords
    outputs:
        index_in_json: index information
        documents: preprocessed recipe ingredients
        word_set: contain all vocabulary
    '''

    data = json.load(open(json_filename))
    documents = []
    index_in_json = []
    title_set = set()
    word_set = set()
    cnt = 0
    len_data = str(len(data))
    cur_customized_stopwords = set()

    for i in range(0, len(data)):
        if len(data[i]) == num_attribute and len(data[i]['ingredients']) != 0:
            if data[i]['title'] not in title_set:

                title_set.add(data[i]['title'])
                index_in_json.append(i)
                ingredients = data[i]['ingredients']

                actual_ingredients = []
                for each_ingredient in ingredients:
                    if is_lower_case:
                        each_ingredient = each_ingredient.lower()

                    tokens = nltk.word_tokenize(each_ingredient)

                    if is_stem:
                        singles = [stemmer.stem(token) for token in tokens]
                        for ele in customized_stopwords:
                            cur_customized_stopwords.add(stemmer.stem(ele))

                    if is_remove_stopwords:
                        filtered_words = [word for word in singles if (word
                            not in stopwords.words('english') and
                            word not in cur_customized_stopwords)]
                    else:
                        filtered_words = singles
                    filtered_words_2 = []
                    if is_remove_puctuation:
                        for word in filtered_words:
                            if word.isalpha():
                                filtered_words_2.append(word)
                                word_set.add(word)
                    else:
                        filtered_words_2 = filtered_words

                    actual_ingredients = actual_ingredients + filtered_words_2
                documents.append(actual_ingredients)



    return index_in_json, documents, word_set


def generate_inverted_index(index_in_json, documents, word_set):
    '''
    This function is used to generate interted index for efficiency
    inputs:
        index_in_json: index information
        documents: preprocessed recipe ingredients
        word_set: contain all vocabulary
    outputs:
        inverted_index: matrix. Contain information about each word
    '''

    N = len(documents)
    num_word = len(word_set)
    cur_word = 0
    inverted_index = {}
    for each_word in word_set:
        inverted_index[each_word] = [0]
        cur_word = cur_word + 1

        for i in range(0, len(documents)):
            index = index_in_json[i]
            document = documents[i]
            if each_word in document:
                inverted_index[each_word][0] = \
                    inverted_index[each_word][0] + 1
                inverted_index[each_word].append((i,
                    document.count(each_word)))
    return inverted_index


def get_document_length(index_in_json, documents, inverted_index):
    '''
    This function is used to calculate length of each document
    inputs:
        index_in_json: index information
        documents: preprocessed recipe ingredients
        inverted_index: matrix. Contain information about each word
    outputs:
        doc_length: vector. length of each document
    '''

    doc_length = []
    for i in range(0, len(documents)):
        document = documents[i]
        length = 0.0
        for each_word in document:
            tf = math.log(document.count(each_word) + 1.0)
            idf = inverted_index[each_word][0]
            length = length + tf*idf*tf*idf
        doc_length.append(math.sqrt(length))
    return doc_length


def doc_ranking(k1, b, query, index_in_json, documents,
    inverted_index, doc_length):
    '''
    This function is used to rank document with BM25 model.
    inputs:
        k1, b: parameters for BM25
        query: given ingredients
        index_in_json: index information
        documents: preprocessed recipe ingredients
        inverted_index: matrix. Contain information about each word
        doc_length: vector. length of each document
    outputs:
        doc_rank: documents ordered by score
    '''

    #preprocessing query
    N = len(documents)
    query_set = set()
    if is_lower_case:
        query = query.lower()
    tokens = nltk.word_tokenize(query)
    if is_stem:
        singles = [stemmer.stem(token) for token in tokens]
    if is_remove_stopwords:
        filtered_words = [word for word in singles \
            if word not in stopwords.words('english')]
    else:
        filtered_words = singles
    filtered_words_2 = []
    if is_remove_puctuation:
        for word in filtered_words:
            if word.isalpha():
                filtered_words_2.append(word)
                query_set.add(word)
    else:
        filtered_words_2 = filtered_words
    average_length = 0.0
    for each_doc in documents:
        average_length = average_length + len(each_doc)
    average_length = float(average_length) / len(documents)
    #doc doc_ranking
    doc_rank = {}
    query_length = 0.0
    for each_word in query_set:
        if each_word in inverted_index:
            for i in range(1, len(inverted_index[each_word])):
                document_index = inverted_index[each_word][i][0]
                document_count = inverted_index[each_word][i][1]
                tf_doc = float(document_count)
                df_doc = inverted_index[each_word][0]
                part_1 = (N - df_doc + 0.5) / float(df_doc + 0.5)
                part_2 = ((k1 + 1) * tf_doc) / \
                    (tf_doc + k1 * (1 - b + b * \
                        (len(documents[document_index]) / average_length)))

                if document_index not in doc_rank:
                    doc_rank[document_index] = 0.0

                doc_rank[document_index] = \
                    doc_rank[document_index] + math.log(part_1 * part_2)
    return doc_rank


def delete_food(sorted_doc_rank, documents, without_food):
    '''
    This function is used to delete the ingredients that
        you do not want to be included in your recipe
    inputs:
        sorted_doc_rank: documents ordered by score
        documents: original documnets
        without_food: what ingredients you do not want
    outputs:
        filtered_doc_index: documents(recipes) ordered by score
    '''

    if len(without_food) == 0:
        filtered_doc_index = []
        for ele in sorted_doc_rank:
            index = ele[0]
            filtered_doc_index.append(index)
    else:
        without_food_set = set()
        if is_lower_case:
            without_food = without_food.lower()
        tokens = nltk.word_tokenize(without_food)
        if is_stem:
            singles = [stemmer.stem(token) for token in tokens]
        if is_remove_stopwords:
            filtered_words = [word for word in singles \
                if word not in stopwords.words('english')]
        else:
            filtered_words = singles
        filtered_words_2 = []
        if is_remove_puctuation:
            for word in filtered_words:
                if word.isalpha():
                    filtered_words_2.append(word)
                    without_food_set.add(word)
        else:
            filtered_words_2 = filtered_words
        filtered_doc_index = []
        for ele in sorted_doc_rank:
            index = ele[0]
            is_valid = True
            for each_no_word in without_food_set:
                if each_no_word in documents[index]:
                    is_valid = False
                    break
            if is_valid:
                filtered_doc_index.append(index)
    return filtered_doc_index


def get_data(json_filename, top_n, doc_index, documents):
    '''
    This function is used to retrieve the whole recipe from json file.
    inputs:
        jason_filename: data
        top_n: how many recipes you want
        doc_index: index information
        documents: preprocessed recipe ingredients
    outputs:
        s: saves recipes info into a list
    '''

    data = json.load(open(json_filename))
    s=[]
    for i in range(0, top_n):
        recipe_info = []
        index = load_index_in_json[doc_index[i]]
        recipe_info.append(data[index]['title'][0:-1])
        ingred = ''
        for i in range(len(data[index]['ingredients'])):
            ingred += data[index]['ingredients'][i] + ", "

        recipe_info.append(ingred)
        direct = ''
        for i in range(len(data[index]['directions'])):
            direct += str(i + 1) + ". " + data[index]['directions'][i] + " "
        recipe_info.append(direct)
        s.append(recipe_info)
    return s


def save_func(filename, data):
    '''
    This function is used to save tmp data
    inputs:
        filename: string
        data: what data you want to save
    '''

    with open(filename, "wb") as fp1:
        pickle.dump(data, fp1)


def load_func(filename):
    '''
    This unction is used to load tmp data
    inputs:
        filename: string
        data: what data you want to load
    '''

    with open(filename, "rb") as fp1:
        data = pickle.load(fp1)
    return data


stemmer = PorterStemmer()
is_lower_case = True
is_stem = True
is_remove_stopwords = True
is_remove_puctuation = True

name_documents = 'documents_BM25'
name_index_in_json = 'index_in_json_BM25'
name_word_set = 'word_set_BM25'
name_inverted_index = 'inverted_index_BM25'
num_attribute = 11
json_filename = 'full_format_recipes.json'
name_doc_length = 'doc_length_BM25'


k1 = 1.1
b = 0.99
customized_stopwords = {"spoon", "cups", "large",
    "teaspoon", "medium", "small", "Freshly", "sheets", "pound",
    "tablespoon", "ounce", "lb"}


load_documents = load_func(name_documents)
load_index_in_json = load_func(name_index_in_json)
load_word_set = load_func(name_word_set)
load_inverted_index = load_func(name_inverted_index)
load_doc_length = load_func(name_doc_length)


def find_recipe(json_filename, query, top_n, without_food,
    load_index_in_json, load_documents, load_inverted_index, load_doc_length):
    '''
    Main Function is used to find recipe we want.
    inputs:
        jason_filename: data
        query: given ingredients
        top_n: how many recipes you want us to return
        without_food: what ingredients you do not want to include
        load_index_in_json: tmp index file
        load_documents: tmp documnets file
        load_inverted_index: saved inverted_index file
        load_doc_length: saved doc length file
    outputs:
        dt: recipes information we want
    '''

    doc_rank = doc_ranking(k1, b, query, load_index_in_json,
        load_documents, load_inverted_index, load_doc_length)
    sorted_doc_rank = sorted(doc_rank.items(),
        key=operator.itemgetter(1), reverse=True)
    data = json.load(open(json_filename))
    for i in range(0, top_n):
        index = load_index_in_json[sorted_doc_rank[i][0]]

    filtered_doc_index = delete_food(sorted_doc_rank,
        load_documents, without_food)

    dt = get_data(json_filename, top_n, filtered_doc_index, load_documents)

    return dt
