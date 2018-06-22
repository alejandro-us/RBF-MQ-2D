#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 19:31:08 2018

@author: alex
"""
import time

def timePass(method):
    '''
    Calculates the elapsed execution time of a function.
    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(method.__name__, ' elapsed time: ', (te - ts), '\n')
        return result

    return timed