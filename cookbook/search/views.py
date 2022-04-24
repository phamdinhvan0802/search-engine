# Original/Heavily modified

import json
import traceback
import sys
import csv
import os

from functools import reduce
from operator import and_

from django.shortcuts import render
from django import forms

from BM25 import *
from Scrapeimage import *

NOPREF_STR = 'No preference'
RES_DIR = os.path.join(os.path.dirname(__file__), '..', 'res')


# create a searchform class with the form fields we need
class SearchForm(forms.Form):
    query = forms.CharField(
        label='Nguyên Liệu:',
        help_text='VD: eggs',
        required=True)

    num = forms.IntegerField(label='Số lượng công thức:', 
                             help_text='VD: 1-10', 
                             min_value=1, max_value=10)
    
    without_food = forms.CharField(label='Thức ăn không mong muốn (Tùy chọn)', 
                                   help_text='VD: onion', 
                                   required=False)


def home(request):
    '''
    Process the request; Extracts user's query and  process it, 
    convert the input query into variables in our existing python 
    function. Combines a variables we used with a given context 
    dictionary and returns an HttpResponse object with that rendered text.
    '''

    result=None
    food_query = None
    img_url = None
    res = None
    num_char = 0
    if request.method == 'GET':
        form = SearchForm(request.GET)
        if form.is_valid():
            food_query = form.cleaned_data['query']
            without = form.cleaned_data['without_food']
            num_char = form.cleaned_data['num']

            result = find_recipe('full_format_recipes.json', food_query, 
                                 num_char, without, load_index_in_json, 
                                 load_documents, load_inverted_index, load_doc_length)

            res = []
            for i in result:
                img_url = extract_images(i[0])
                res.append((i[0], i[1], i[2], img_url))

           
    else:
        form = SearchForm()
    
    return render(request, 'search/index.html', {'form': form,"result": result, 
                                                 "food_query": food_query, 
                                                 'res': res, 'num_char': num_char})



