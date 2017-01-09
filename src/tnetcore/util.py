'''
Created on Feb 18, 2015

@author: wohlhart
'''
from __future__ import print_function

def readCfgParam(cfg,section,key,default):
    if cfg.has_option(section,key):
        return cfg.get(section,key)
    else:
        return default
        
def readCfgIntParam(cfg,section,key,default):    
    if cfg.has_option(section,key):
        return cfg.getint(section,key)
    else:
        return default

def readCfgFloatParam(cfg,section,key,default):    
    if cfg.has_option(section,key):
        return cfg.getfloat(section,key)
    else:
        return default
    
def readCfgBooleanParam(cfg,section,key,default):    
    if cfg.has_option(section,key):
        return cfg.getboolean(section,key)
    else:
        return default    
    
def intOrNone(arg):
    try:
        return int(arg)
    except ValueError:
        return None
    
def readCfgIntNoneListParam(cfg,section,key,default):    
    if cfg.has_option(section,key):
        strVal = cfg.get(section,key)
        l = strVal.split(',')
        l = map(intOrNone,l)
        return l
    else:
        return default    

def readCfgStrListParam(cfg,section,key,default):    
    if cfg.has_option(section,key):
        val = cfg.get(section,key)
        return [x.strip() for x in val.split(',')]
    else:
        return default    

