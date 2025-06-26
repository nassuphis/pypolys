#######################################
# poly solver state
#######################################

import sys
import json
import ast
import functools
import numpy as np
import argparse
from . import registry
from . import polyutil as pu


def sample(t0,t1):
    xfrm  = flow['xfrm']
    poly  = flow['poly']
    zfrm  = flow['zfrm']
    solve = flow['solve']
    f1  = lambda acc, func: func(acc[0], acc[1]) 
    tn  = functools.reduce(f1,xfrm,(t0,t1))
    pcf = poly(tn[0], tn[1])
    f2  = lambda acc, func: func(acc)
    zcf = functools.reduce(f2, zfrm, pcf)
    roots = solve(zcf)
    return roots

#######################################
# Poly definition
#######################################
poly = { # define the poly
    'name'        : 'poly', # dict name          
    'n'           : None,
    'degree'      : None,
    'xfrm'        : None,
    'poly'        : None,
    'zfrm'        : None,
    'solve'       : None,
    'args.seti'   : None,
    'args.setf'   : None,
    'args.setl'   : None,
    'args.sets'   : None,
    'args.levels' : None,
    'levels'      : None
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
    'chunk'   : 0,
    'samples' : 100_000
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
    'view'   : (-1-1j,1+1j),
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
    'rotate'   : 0,
    'args.png' : None
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
# export state in 3 forms:
# 1. as dictionary
# 2. as json
# 3. as command line
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
    config['view']['view'] = f"{config['view']['view']}".replace(' ','')
    return config


def state2json():
    config = state2dict()
    json_state = json.dumps(config, indent=4)
    return json_state

def state2cjson():
    config = state2dict()
    json_state = json.dumps(config,separators=(",", ":"))
    return json_state

def state2cli():

    args = [ data['stem'] or '']

    def add_arg(flag, value):
        if value is None: 
            return
        if isinstance(value, bool):    
            if value: args.extend([flag])
            return
        if value=='':    
            return
        args.extend([flag, str(value).replace(' ', '')])

    add_arg('--mode',data['mode'])
    add_arg('--verbose',job['verbose'])
    add_arg('--roots',job['roots']/ 1_000_000)
    add_arg('--procs',job['procs'])
    add_arg('--alpha',view['alpha'])
    add_arg('--margin',view['margin'])
    add_arg('--res',view['res'])
    add_arg('--samples',job['samples'])
    add_arg('--view',view['subview'])
    add_arg('--seti',poly['args.seti'])
    add_arg('--setf',poly['args.setf'])
    add_arg('--setl',poly['args.setl'])
    add_arg('--levels',poly['args.levels'])
    add_arg('--xfrm', poly['xfrm'])
    add_arg('--poly',poly['poly'])
    add_arg('--zfrm',poly['zfrm'])
    add_arg('--solve',poly['solve'])
    add_arg('--rotate',png['rotate'])
    add_arg('--png',png['args.png'])

    cli = ' '.join(str(x) for x in args)

    return cli

#######################################
# import state in 3 forms:
# 1. as dictionary
# 2. as json
# 3. as command line
#######################################

def dict2state(config):
    if isinstance(config.get('poly'),dict):
        poly.update(config.get('poly'))
    if isinstance(config.get('data'),dict):
        data.update(config.get('data') or {})
    if isinstance(config.get('job'),dict):
        job.update(config.get('job') or {})
    if isinstance(config.get('view'),dict):
        view.update(config.get('view') or {})
    if isinstance(config.get('png'),dict):
        png.update(config.get('png') or {})
    check_state()
    
    flow['xfrm']  = pu.get_function_vector(registry.xfun, poly.get('xfrm'))
    flow['poly']  = pu.get_function(registry.pfun,poly.get('poly'))
    flow['zfrm']  = pu.get_function_vector(registry.zfun, poly.get('zfrm'))
    flow['solve'] = pu.get_function(registry.sfun, poly.get('solve'))

    job['chunk'] = int(job['roots']) // (poly.get('degree') or 10) // (job.get('procs') or 16)

    poly.update(pu.seti(poly.get("args.seti")))
    poly.update(pu.setf(poly.get("args.setf")))
    poly.update(pu.sets(poly.get("args.sets")))
    poly.update(pu.setl(poly.get("args.setl")))

    n = poly.get("n") or 10
    poly["cf_start"]=pu.cvec2json(np.poly(pu.random_coeff(n)))
    poly["cf_end"]=pu.cvec2json(np.poly(pu.random_coeff(n)))
    
    if isinstance(view.get("view"),str):
        view['view']=ast.literal_eval(view['view'])

    if poly.get('args.levels') is not None: 
        poly['levels']= sorted(map(float, poly['args.levels'].split(',')))

    if job.get("args.roots") is not None: 
        job['roots']  = job['args.roots'] * 1_000_000

    return

def json2state(js):
    config = json.loads(js)
    dict2state(config)
    return

def cli2state(cli):
    parser = argparse.ArgumentParser(description="root locus")
    parser.add_argument('stem',nargs="?",type=str, default=None, help="stem")
    parser.add_argument('-m','--mode', choices=["write","show","showinit","png"],default=None,help="mode")
    parser.add_argument('--verbose',action='store_true',help="verbose")
    parser.add_argument('--roots', type=float, default=None, help="roots")
    parser.add_argument('--procs', type=int, default=None, help="processes")
    parser.add_argument('--alpha', type=float, default=None, help="visible quantile")
    parser.add_argument('--margin', type=float, default=None, help="estimation margin")
    parser.add_argument('-r','--res', type=int, default=None, help=".png resolution")
    parser.add_argument('--samples', type=int, default=None, help="range estimation sample count")
    parser.add_argument('-v','--view', type=str, default=None, help="subview selection")
    parser.add_argument('--seti', type=str, default=None, help="seti")
    parser.add_argument('--setf', type=str, default=None, help="setf")
    parser.add_argument('--sets', type=str, default=None, help="sets")
    parser.add_argument('--setl', type=str, default=None, help="setl")
    parser.add_argument('--levels', type=str, default=None, help="levels")
    parser.add_argument('-x', '--xfrm', type=str, default=None, help="xfrm")
    parser.add_argument('-p', '--poly', type=str, default=None, help="poly")
    parser.add_argument('-z', '--zfrm', type=str, default=None, help="zfrm")
    parser.add_argument('-s', '--solve', type=str, default=None, help="solve")
    parser.add_argument('--rotate', type=int, default=None, help="rotate")
    parser.add_argument('--png', type=str, default=None,help="invert,blurr,add,subtract")
    
    args = parser.parse_args(cli.split())

    job['verbose'] = args.verbose

    pu.ns2dict_required(args,data,"stem")
    pu.ns2dict_required(args,data,"mode")
    pu.ns2dict_required(args,poly,"xfrm")
    pu.ns2dict_required(args,poly,"poly")
    pu.ns2dict_required(args,poly,"zfrm")
    pu.ns2dict_required(args,poly,"solve")
    pu.ns2dict_required(args,view,"samples")
    pu.ns2dict_required(args,view,"alpha")
    pu.ns2dict_required(args,view,"margin")
    pu.ns2dict_required(args,view,"res")
    pu.ns2dict_required2(args,view,"view","subview")
    pu.ns2dict_required(args,png,"rotate")
    pu.ns2dict_optional(args,png,"png")
    pu.ns2dict_optional2(args,png,"png","args.png")
    pu.ns2dict_required(args,job,"procs")
    
    pu.ns2dict_optional2(args,poly,"seti","args.seti")
    pu.ns2dict_optional2(args,poly,"setf","args.setf")
    pu.ns2dict_optional2(args,poly,"sets","args.sets")
    pu.ns2dict_optional2(args,poly,"setl","args.setl")
    pu.ns2dict_optional2(args,poly,"levels","args.levels")
    pu.ns2dict_optional2(args,job,"roots","args.roots")
    

    dict2state({})
    return 

