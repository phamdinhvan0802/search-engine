## Original. NDCG formula learned from Järvelin, K., & Kekäläinen, J. (2002). 
## Cumulated gain-based evaluation of IR techniques. ACM Transactions on 
## Information Systems, 20(4), 422-446.
## This file contains code for evaluating vector space model and BM25 model
## based on normalized discounted cumulative gain score.


import json
import nltk
import math
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *
import csv
import pickle
from sklearn import preprocessing
import operator
from collections import OrderedDict


def read_and_preprocessing(json_filename, num_attribute,
    is_lower_case, is_stem,is_remove_stopwords,
    is_remove_puctuation, stemmer, if_remove_special):
    '''
    This function is used to preprocessing json data and only remain 
    ingredients
    inputs:
        jason_filename: data
        num_attribute: set to 11 at the end to include recipe
            w/ 11 attributes (title, ingredients, etc.)
        is_lower_case: flag, logical
        is_stem,is_remove_stopwords: flag, logical
        is_remove_puctuation: flag, logical
        stemmer: an object for stemming, returned from PorterStemmer,
            which is a function in nltk.stem.Porter
        if_remove_special: logical to denote remove special word or not
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

    for i in range(0, len(data)):
        if len(data[i]) == num_attribute and len(data[i]['ingredients']) != 0:
            if data[i]['title'] not in title_set:
                print(str(i) + '/' + len_data)
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

                    if is_remove_stopwords:
                        filtered_words = [word for word in singles
                            if word not in stopwords.words('english')]

                    else:
                        filtered_words = singles

                    special_stop_words = set(['cup', 'large', 'small',
                        'medium', 'pound', 'teaspoon',
                        'tablespoon', 'tablespoons',
                        'cups', 'pounds', 'lb', 'chopped',
                        'thick-sliced', 'ripe', 'pitted',
                        'peeled', 'divided', 'sliced',
                        'ounce','ounces', 'fresh'])

                    if if_remove_special:
                        filtered_words = [word for word in filtered_words
                            if word not in special_stop_words]

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


def generate_inverted_index(index_in_json, documents, word_set, if_tfidf):
    '''
    This function is used to generate interted index for efficiency
    inputs:
        index_in_json: index information
        documents: preprocessed recipe ingredients
        word_set: contain all vocabulary
        is_stem,is_remove_stopwords: flag, logical
        if_tfidf: flag, locical. To denote if use tfidf model
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
        print(cur_word, num_word)

        for i in range(0, len(documents)):
            index = index_in_json[i]
            document = documents[i]
            if each_word in document:
                inverted_index[each_word][0] = \
                    inverted_index[each_word][0] + 1
                inverted_index[each_word].append((i,
                    document.count(each_word)))
        if if_tfidf is True:
            inverted_index[each_word][0] = 1.0 + \
                math.log(float(N) / float(inverted_index[each_word][0]))
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
        print(i,len(documents))
        for each_word in document:
            tf = math.log(document.count(each_word) + 1.0)
            idf = inverted_index[each_word][0]
            length = length + tf*idf*tf*idf
        doc_length.append(math.sqrt(length))
    return doc_length


def doc_ranking_BM25(k1, b, query, index_in_json, documents,
    inverted_index, doc_length):
    '''
    This unction is used to rank document with BM25 model.
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


def doc_ranking(query, index_in_json, documents, inverted_index, doc_length):
    '''
    This function is used to rank document with tfidf model 
    (vector space model).
    inputs:
        query: given ingredients
        index_in_json: index information
        documents: preprocessed recipe ingredients
        inverted_index: matrix. Contain information about each word
        doc_length: vector. length of each document
    outputs:
        doc_rank: documents ordered by score
    '''

    #preprocessing query
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
    doc_rank = {}
    query_length = 0.0
    for each_word in query_set:
        if each_word in inverted_index:
            tf_query = math.log(filtered_words_2.count(each_word) + 1.0)
            idf_query = inverted_index[each_word][0]
            query_word_weight = tf_query * idf_query
            query_length = query_length + query_word_weight*query_word_weight
            for i in range(1, len(inverted_index[each_word])):
                document_index = inverted_index[each_word][i][0]
                document_count = inverted_index[each_word][i][1]
                tf_doc = math.log(document_count + 1.0)
                idf_doc = inverted_index[each_word][0]
                doc_word_weight = tf_doc * idf_doc
                if document_index not in doc_rank:
                    doc_rank[document_index] = 0.0
                doc_rank[document_index] = doc_rank[document_index] + \
                    doc_word_weight * query_word_weight

    query_length = math.sqrt(query_length)
    for document_index in doc_rank:
        doc_rank[document_index] = float(doc_rank[document_index]) / \
            (query_length * doc_length[document_index])
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


def get_data(json_filename, top_n, doc_index, documents,
    load_index_in_json, result_list, title_list):
    '''
    This function is used to retrieve the whole recipe from json file.
    inputs:
        jason_filename: data
        top_n: how many recipes you want
        doc_index: index information
        documents: preprocessed recipe ingredients
        result_list, title_list: vector contain result and title
    '''


    data = json.load(open(json_filename))
    to_print ={}
    for i in range(0, top_n):
        index = load_index_in_json[doc_index[i]]
        to_print['title'] = data[index]['title']
        to_print['ingredients'] = documents[doc_index[i]]

        title_list.append(data[index]['title'])
        result_list.append(documents[doc_index[i]])


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
    This function is used to load tmp data
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
final_result = []
final_titles = []
k1 = 1.1
b = 0.99


query_list = ['beef, potato, onion, carrot, black pepper',
    'tomato, egg, tuna, potato, olive',
    'broccoli, pepper, carrot, onion',
    'shrimp, garlic, wine, butter',
    'chicken breast, tomato, black pepper, mushroom',
    'salmon, lemon, olive oil',
    'red curry, chicken broth, shrimp','chicken, soy sauce, vinegar, sugar',
    'chicken breast, hot chile paste, sugar, vinegar, peanuts',
    'pork spareribs, honey, catchup']


def LTfunc(if_remove_special, suffix, if_preprocess, if_tfidf):
    '''
    This unction is used to evaluation(rank document for specified query).
    inputs:
        if_remove_special: flag, if remove special word or not
        suffix: string
        if_preprocess: flag, if do preprocss part. If not, then load data
        documents: preprocessed recipe ingredients
        if_tfidf: which model you want to use
    '''

    name_documents = 'documents' + suffix
    name_index_in_json = 'index_in_json' + suffix
    name_word_set = 'word_set' + suffix
    name_inverted_index = 'inverted_index' + suffix
    num_attribute = 11
    json_filename = 'full_format_recipes.json'
    name_doc_length = 'doc_length' + suffix

    if if_preprocess is True:
        index_in_json, documents, word_set = \
            read_and_preprocessing(json_filename,
                num_attribute, is_lower_case,
                is_stem,is_remove_stopwords, is_remove_puctuation,
                stemmer, if_remove_special)

        save_func(name_documents, documents)
        save_func(name_index_in_json, index_in_json)
        save_func(name_word_set, word_set)

    load_documents = load_func(name_documents)
    load_index_in_json = load_func(name_index_in_json)
    load_word_set = load_func(name_word_set)

    if if_preprocess is True:
        inverted_index = generate_inverted_index(load_index_in_json,
            load_documents, load_word_set, if_tfidf)

        save_func(name_inverted_index, inverted_index)

    load_inverted_index = load_func(name_inverted_index)

    if if_preprocess is True:
        doc_length = get_document_length(load_index_in_json, \
            load_documents, load_inverted_index)
        save_func(name_doc_length, doc_length)
    load_doc_length = load_func(name_doc_length)

    result_list_list = []
    title_list_list = []
    for query in query_list:
        doc_rank = 0
        if if_tfidf is True:
            doc_rank = doc_ranking(query, load_index_in_json, \
                load_documents, load_inverted_index, load_doc_length)
        else:
            doc_rank = doc_ranking_BM25(k1, b, query, load_index_in_json, \
                load_documents, load_inverted_index, load_doc_length)
        sorted_doc_rank = sorted(doc_rank.items(),
            key=operator.itemgetter(1), reverse=True)

        without_food = 'beet'
        filtered_doc_index = delete_food(sorted_doc_rank,
            load_documents, without_food)

        result_list = []
        title_list = []
        top_n = 1000
        get_data(json_filename, top_n, filtered_doc_index,
            load_documents, load_index_in_json, result_list, title_list)
        print_info = "processing " + query
        print(print_info)

        result_list_list.append(result_list)
        title_list_list.append(title_list)

    final_result.append(result_list_list)
    final_titles.append(title_list_list)


def query_preprocess(query):
    '''
    This function is used to preprocess query.
    inputs:
        query: given ingredients
    output:
        filtered_words_2: query after processed
    '''

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
    return filtered_words_2


def calculate_score(query, ingredients_list):
    '''
    This function is used to calculate query score.
    inputs:
        query: given ingredients
        ingredients_list: list of ingredients
    output:
        result: score of query with ingredients
    '''

    result = []
    for ingredients in ingredients_list:
        val = 0
        for word in query:
            if word in ingredients:
                val += 1
        result.append(val)
    return result


def calculate(atX, list):
    '''
    This function is used to calculate doc score.
    inputs:
        atX: top what
        list: list of ingredients
    output:
        val: score of query with ingredients
    '''

    val = list[0]
    for i in range(atX):
        if i == 0:
            continue
        val += list[i] / math.log(i + 1, 2)
    return val


def dcg(atX):
    '''
    This function is used to calculate NDCG value.
    inputs:
        atX: top what
    '''

    for j in range(len(query_list)):
        query = query_preprocess(query_list[j])
        ideal_titles = []
        ideal_ingredients = []
        scores = []
        for i in range(len(final_titles)):
            cur_titles = final_titles[i][j]
            cur_ingredients = final_result[i][j]
            scores.append(calculate_score(query, cur_ingredients))
            for k in range(len(cur_titles)):
                title = cur_titles[k]
                if title not in set(ideal_titles):
                    ideal_titles.append(title)
                    ideal_ingredients.append(cur_ingredients[k])

        ideal_score = sorted(calculate_score(query, ideal_ingredients),
            reverse = True)
        idcg = calculate(atX, ideal_score) + 16
        for score in scores:
            print(str(calculate(atX, score) / idcg)+ ",",)
        print("")


use_tfidf = True
LTfunc(True, "_tfidf_removed_stop_words", False, use_tfidf)

use_tfidf = False
LTfunc(True, "_BM25_removed_stop_words", False, use_tfidf)

dcg(100)
