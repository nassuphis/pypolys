#!/usr/bin/env python

import numpy as np
import json
import re
import numexpr as ne

#######################################
# random coefficient
#######################################

def random_coeff(n: int) -> np.ndarray:
    real = np.random.uniform(-1.0, 1.0, n)   # real parts
    imag = np.random.uniform(-1.0, 1.0, n)   # imaginary parts
    return real + 1j * imag

#######################################
# complex vector to json and back
#######################################

def cvec2json(vec: np.ndarray) -> str:
    """Convert a 1D complex numpy array to a JSON string as list of [real, imag]."""
    data = [[float(x.real), float(x.imag)] for x in vec]
    return json.dumps(data)

def json2cvec(s: str) -> np.ndarray:
    """Convert a JSON string (list of [real, imag]) back to a complex numpy array."""
    data = json.loads(s)
    return np.array([complex(real, imag) for real, imag in data], dtype=complex)

#######################################
# extract function from dictionary
#######################################
def get_function(dict,fn):
    if fn is None:
        print(f"get_function: fn is None")
        return None
    if fn not in dict:
        raise KeyError(f"get_function: '{fn}' not in dict")
    value = dict[fn]
    if not callable(value):
        raise KeyError(f"get_function: dictionary value '{fn}' is not callable.")
    return value

#######################################
# extract function vector from dictionary
#######################################
def get_function_vector(dictionary, functions):
    selected_functions = []
    if functions is None: 
        return selected_functions
    if functions.strip()=='': 
        return selected_functions
    function_names = [name.strip() for name in functions.split(',')]

    for name in function_names:
        if name not in dictionary:
            raise KeyError(f"get_function_vector: '{name}' not in dict")
        value = dictionary[name]
        if not callable(value):
            raise ValueError(f"get_function_vector: dict value '{name}' not callable")
        selected_functions.append(value)

    return selected_functions

#######################################
# string to dictionary of strings
#######################################
def sets(param_string):
    param_dict = {}
    if param_string is None:
        return param_dict
    if '=' not in param_string: 
        return param_dict
    pattern = r'([^=,]+)=(?:%([^%]*)%|([^,]+))'
    matches = re.findall(pattern, param_string)
    for key, percent_value, unquoted_value in matches:
        value = percent_value if percent_value else unquoted_value
        param_dict[key.strip()] = value.strip()
    return param_dict

#######################################
# string to dictionary of logicals
#######################################
def setl(arg):
    true_values = {"true", "t", "yes", "y", "1"}
    state = {}
    if arg is None:
        return state
    if '=' not in arg:
        return state    
    pairs = arg.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        state[key] = value in true_values
    return state

#######################################
# string to dictionary of floats
#######################################
def setf(arg):
    state = {}
    if arg is None:
        return state
    if '=' not in arg:
        return state
    pairs = arg.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        try:
            state[key] = ne.evaluate(value)  # Convert values to floats
        except ValueError:
            raise ValueError(f"Invalid value for {key}: {value} (must be a float)")
    return state

#######################################
# string to dictionary of integers
#######################################
def seti(arg):
    state = {}
    if arg is None:
        return state
    if '=' not in arg:
        return state   
    pairs = arg.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        try:
            state[key] = int(value)  # Convert values to integers
        except ValueError:
            raise ValueError(f"Invalid value for {key}: {value} (must be an integer)")
    return state


#######################################
# string to dictionary of integers
#######################################

def ns2dict_optional(ns,dict,key):
    if getattr(ns,key) is not None: 
        dict[key] = getattr(ns,key)

def ns2dict_required(ns,dict,key):
    ns2dict_optional(ns,dict,key)
    if dict.get(key) is None:
        raise KeyError(f"{key} not in dict")

def ns2dict_optional2(ns,dict,nkey,dkey):
    if getattr(ns,nkey) is not None: 
        dict[dkey] = getattr(ns,nkey) 

def ns2dict_required2(ns,dict,nkey,dkey):
    ns2dict_optional2(ns,dict,nkey,dkey)
    if dict[dkey] is None:
        raise KeyError(f"{dkey} not in dict")
