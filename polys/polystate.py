#######################################
# poly solver state
#######################################
import sys
import io
import pprint
import re
from . import xfrm as xfrm_module
from . import zfrm as zfrm_module
from . import solve as solve_module
import json
import ast
import numpy as np


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
# Poly definition
#######################################
poly = { # define the poly
    'name'    : 'poly', # dict name          
    'n'       : None,
    'degree'  : None,
    'xfrm'    : None,
    'poly'    : None,
    'zfrm'    : None,
    'solve'   : None
}

def check_poly():
    req = ['name','xfrm', 'poly', 'zfrm', 'solve','degree','n']
    missing_keys = [key for key in req if key not in poly]
    if missing_keys:
        print(f"State: poly is missing the following keys: {', '.join(missing_keys)}")
        sys.exit(1)

#######################################
# Flow
#######################################
flow = { # poly functions
    'name'    : 'flow', # dict name 
    'xfrm'    : None,
    'poly'    : None,
    'zfrm'    : None,
    'solve'   : None
}

def check_flow():
    req = ['name','xfrm', 'poly', 'zfrm', 'solve']
    missing_keys = [key for key in req if key not in flow]
    if missing_keys:
        print(f"State: flow is missing the following keys: {', '.join(missing_keys)}")
        sys.exit(1)


#######################################
# Data definition
#######################################
data = { # define data 
    'name'     :  'data', # dict name 
    'stem'     :  None,
    'mode'     :  'png'
}

def check_data():
    req = ['name','stem', 'mode']
    missing_keys = [key for key in req if key not in data]
    if missing_keys:
        print(f"State: data is missing the following keys: {', '.join(missing_keys)}")
        sys.exit(1)

def fn(stub):
    if data['mode']=="write":
        return f"{data['stem']}/{data['stem']}"+stub
    if data['mode']=="png":
        return f"{data['stem']}"+stub
    print(f"Invalid output mode: {data['mode']}")
    sys.exit(1)

#######################################
# Job definition
#######################################
job = { 
    'name'    : 'job', # dict name 
    'verbose' : True,
    'procs'   : 14,
    'roots'   : 1_000_000,
    'chunk'   : 0
}

def check_job():
    req = ['name','verbose','procs', 'roots', 'chunk']
    missing_keys = [key for key in req if key not in job]
    if missing_keys:
        print(f"State: job is missing the following keys: {', '.join(missing_keys)}")
        sys.exit(1)

#######################################
# View definition
#######################################
view = { # define the view
    'name'   : 'view', # dict name 
    'alpha'  : 0.000,
    'margin' : 0.05,
    'view'   : (0+0j,1+1j),
    'subview': 'full',
    'res'    : 50_000,
    'samples' : 100_000,
}

def check_view():
    req = ['name','alpha','margin', 'view', 'subview', 'res', 'samples']
    missing_keys = [key for key in req if key not in view]
    if missing_keys:
        print(f"State: view is missing the following keys: {', '.join(missing_keys)}")
        sys.exit(1)

#######################################
# PNG generation
#######################################
png = { 
    'name'     : 'png', # dict name 
    'png'      : None,  # add, subtract, blurr, rotate
    'rotate'   : 0
}

def check_png():
    req = ['name','png','rotate']
    missing_keys = [key for key in req if key not in png]
    if missing_keys:
        print(f"State: png is missing the following keys: {', '.join(missing_keys)}")
        sys.exit(1)

def opng(op):
    return 'png' in png and op in png['png'].split(',')

#######################################
# check state
#######################################

def check_state():
    check_poly()
    check_flow()
    check_data()
    check_job()
    check_view()
    check_png()


#######################################
# update from "k1=v1,k2=v2" str of int
#######################################
def update_int(arg,dictionary):
    if '=' not in arg:
        return
    state = {}
    pairs = arg.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        try:
            state[key] = int(value)  # Convert values to integers
        except ValueError:
            raise ValueError(f"Invalid value for {key}: {value} (must be an integer)")
        
    n = state.get("n") or 10
    state["cf_start"]=cvec2json(np.poly(random_coeff(n)))
    state["cf_end"]=cvec2json(np.poly(random_coeff(n)))
    dictionary.update(state)

#######################################
# update from "k1=v1,k2=v2" str of flt
#######################################
def update_flt(arg,dictionary):
    if '=' not in arg:
        return
    state = {}
    pairs = arg.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        try:
            state[key] = float(value)  # Convert values to floats
        except ValueError:
            raise ValueError(f"Invalid value for {key}: {value} (must be a float)")
    dictionary.update(state)


#######################################
# update from "k1=v1,k2=v2" str of str
#######################################

def sets(param_string):
    pattern = r'([^=,]+)=(?:%([^%]*)%|([^,]+))'
    matches = re.findall(pattern, param_string)
    param_dict = {}
    for key, percent_value, unquoted_value in matches:
        value = percent_value if percent_value else unquoted_value
        param_dict[key.strip()] = value.strip()
    return param_dict

def update_str(arg,dictionary):
    if '=' not in arg: return
    state = sets(arg)
    dictionary.update(state)

#######################################
# update from "k1=true,k2=false" str of lgl
#######################################
def update_lgl(arg,dictionary):
    true_values = {"true", "t", "yes", "y", "1"}
    if '=' not in arg:
        return
    state = {}
    pairs = arg.split(',')
    for pair in pairs:
        key, value = pair.split('=')
        state[key] = value in true_values
    dictionary.update(state)

#######################################
# update from "k1=v1,k2=v2" str of str
#######################################
def update_levels(arg,dictionary):
    vals = map(float, arg.split(','))
    levels = sorted(vals)
    dictionary['levels']= levels

#######################################
# extract function from module
#######################################
def get_function(module,function):
    if not hasattr(module, function):
        print(f"State: '{function}' not in module.")
        sys.exit(1)
    value = getattr(module, function)
    if not callable(value):
        print(f"State: module value '{function}' is not callable.")
        sys.exit(1)
    return value

#######################################
# extract function vector from module
#######################################
def get_function_vector(module, functions):
    function_names = [name.strip() for name in functions.split(',')]
    selected_functions = []
    for name in function_names:
        if not hasattr(module, name):
            print(f"State: '{name}' not in module.")
            sys.exit(1)
        value = getattr(module, name)
        if not callable(value):
            print(f"State: module value '{name}' is not callable.")
            sys.exit(1)
        selected_functions.append(value)   
    return selected_functions


#######################################
# update flow dictionary
#######################################
def update_flow():
    check_poly()
    flow['xfrm']  = get_function_vector(xfrm_module, poly['xfrm'])
    flow['poly']  = get_function(poly_module, poly['poly'])
    flow['zfrm']  = get_function_vector(zfrm_module, poly['zfrm'])
    flow['solve'] = get_function(solve_module, poly['solve'])

#######################################
# dictionary -> python code
#######################################
def export(dictionary,name):
    output = io.StringIO()
    printer = pprint.PrettyPrinter(indent=4, width=80, compact=False)
    pretty_dict = printer.pformat(dictionary)
    pretty_dict = "{\n " + pretty_dict[1:-1] + "\n}"
    output.write(f"{name} = " + pretty_dict + "\n")
    return output.getvalue()

#######################################
# dump state as python code
#######################################
def state():
    comment = "#######################################\n"
    state = comment
    state = state + export(poly,"poly")
    state = state + comment
    state = state + export(view,"view")
    state = state + comment
    state = state + export(data,"data")
    state = state + comment
    state = state + export(job,"job")
    state = state + comment
    state = state + export(png,"png")
    return state

#######################################
# dump state as json
#######################################
def state2json():
    config = {
        'poly': poly,
        'flow': {
            'name'    : 'flow', # dict name 
            'xfrm'    : 'none',
            'poly'    : 'none',
            'zfrm'    : 'none',
            'solve'   : 'none'  
        },
        'data': data,
        'job' : job,
        'view': view, 
        'png' : png,
    }
    config['view']['view'] = f"{config['view']['view']}"
    state = json.dumps(config, indent=4)

    return state

#######################################
# dump state as compressed json
#######################################
def state2cjson():
    config = {
        'poly': poly,
        'flow': {
            'name'    : 'flow', # dict name 
            'xfrm'    : 'none',
            'poly'    : 'none',
            'zfrm'    : 'none',
            'solve'   : 'none'  
        },
        'data': data,
        'job' : job,
        'view': view, 
        'png' : png,
    }
    config['view']['view'] = f"{config['view']['view']}"
    state = json.dumps(config,separators=(",", ":"))

    return state

#######################################
# dump state into dict
#######################################
def state2dict():
    config = {
        'poly': poly,
        'flow': {
            'name'    : 'flow', # dict name 
            'xfrm'    : 'none',
            'poly'    : 'none',
            'zfrm'    : 'none',
            'solve'   : 'none'  
        },
        'data': data,
        'job' : job,
        'view': view, 
        'png' : png,
    }
    return config

#######################################
# convert module to dictionary python
#######################################
def pymod2pydict(pys):
    tpys = re.sub(r'^((poly|view|data|job|png))\s*=\s*', r"'\1': ", pys, flags=re.MULTILINE)
    tpys = re.sub(r'^}', r"},", tpys, flags=re.MULTILINE)
    wtpys = '{' + tpys + '}'
    return wtpys


#######################################
# dump state as json
#######################################
def state2dict_code():
    return pymod2pydict(state())

#######################################
# update from dict
#######################################
def dict_update(config):
    poly.update(config['poly'])
    data.update(config['data'])
    job.update(config['job'])
    view.update(config['view'])
    png.update(config['png'])
    job['chunk'] = int(job['roots']) // poly['degree'] // job['procs']
    check_state()
    update_flow()

#######################################
# update from json string
#######################################
def json_update(js):
    config = json.loads(js)
    config['view']['view']=ast.literal_eval(config['view']['view'])
    poly.update(config['poly'])
    data.update(config['data'])
    job.update(config['job'])
    view.update(config['view'])
    png.update(config['png'])
    job['chunk'] = int(job['roots']) // poly['degree'] // job['procs']
    check_state()
    update_flow()

#######################################
# update from python module code
#######################################
def module_code_update(pys):
    config = ast.literal_eval(pymod2pydict(pys))
    poly.update(config['poly'])
    data.update(config['data'])
    job.update(config['job'])
    view.update(config['view'])
    png.update(config['png'])
    job['chunk'] = int(job['roots']) // poly['degree'] // job['procs']
    check_state()
    update_flow()

#######################################
# update from python dict code
#######################################
def dict_code_update(pys):
    config = ast.literal_eval(pys)
    poly.update(config['poly'])
    data.update(config['data'])
    job.update(config['job'])
    view.update(config['view'])
    png.update(config['png'])
    job['chunk'] = int(job['roots']) // poly['degree'] // job['procs']
    check_state()
    update_flow()

