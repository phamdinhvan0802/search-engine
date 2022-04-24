# Modified
# The codes below are "modified" and learned online from https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57

import argparse
import json
import itertools
import logging
import re
import os
import uuid
import sys
import urllib
from urllib.request import urlopen, Request
import requests

from bs4 import BeautifulSoup

def configure_logging():
    '''
    create a logger
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('[%(asctime)s %(levelname)s %(module)s]: %(message)s'))
    logger.addHandler(handler)
    return logger

logger = configure_logging()

REQUEST_HEADER = {
    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}

def get_query_url(query):
    '''
    get url from a input query using encode specific for google
    Input:
       query: string
    '''
    qq = urllib.parse.quote(query)
    return "https://www.google.co.in/search?q=%s&source=lnms&tbm=isch" % qq

def get_soup(url, header):
    '''
    get soup from an url and header
    Input: url, header
    Return: soup object
    '''
    response = urlopen(Request(url, headers=header))
    return BeautifulSoup(response, 'html.parser')

def extract_images_from_soup(soup):
    '''
    extract image from the soup
    '''
    image= soup.find_all("div", {"class": "rg_meta"})
    metadata_dict = (json.loads(e.text) for e in image)
    link_type_records = ((d["ou"], d["ity"]) for d in metadata_dict)
    return link_type_records

def extract_images(query):
    '''
    extract image given an input query 
    and return the url of this image
    '''
    url = get_query_url(query)
    logger.info("Souping")
    soup = get_soup(url, REQUEST_HEADER)
    logger.info("Extracting image urls")
    link_type_records = extract_images_from_soup(soup)

    # we tried many different queries, and we found that the first image is 
    # sometimes not related to food, such as vanilla snow, which is
    # also a song, so we decide to scrape the second image of each recipe
    return list(itertools.islice(link_type_records, 2))





