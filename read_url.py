#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 17:45:33 2017

read url

@author: stepan
"""

import urllib


def read_url(url):
    data = urllib.request.urlopen(url)
    text = data.read().decode('utf-8')
    data.close()
    return text
    
def main():
    read_url('https://github.com/')   
    
if __name__ == "__main__":
    main()