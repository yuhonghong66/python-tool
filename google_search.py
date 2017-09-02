#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 15:56:07 2017

Get URL from google search-engine

@author: stepan
"""

# -*- coding: utf-8 -*-
from google import search

def google_search(query, limit=10):
    urls = []
    print("search start")
    for url in search(query, lang="jp", num=10, stop=1):
        urls.append(url)
    print("search end")
    print(urls)

def main():
    google_search("<search words>")

if __name__ == '__main__':
    main()