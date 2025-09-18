import numpy as np
from numba import njit, types
from numba.typed import Dict

# parameter dict
param = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.Tuple((
        types.unicode_type, 
        types.int8, 
        types.complex128
    )),
)
param["op1"] = ("add",types.int8(1),1.0+1j)
param["op2"] = ("sub",types.int8(1),1.0+1j)
param["op3"] = ("add",types.int8(1),1.0+1j)

state = Dict.empty(
    key_type=types.int8,
    value_type=types.complex128[:],
)
state[types.int8(0)] = np.array([3+3j, 2+2j, 1+1j],dtype=np.complex128)
state[types.int8(1)] = np.array([1+1j, 2+2j, 3+3j],dtype=np.complex128)
state[types.int8(2)] = np.array([3+3j, 3+3j, 3+3j],dtype=np.complex128)

@njit
def op_state(s,p):
    opcode, opkey, opval = param[p]
    res = s[opkey]
    if opcode=="add":
        res += opval
    elif opcode=="sub":
        res -= opval
    elif opcode=="mult":
        res *= opval
    elif opcode=="div":
        res /= opval
    return 


print(f"initial {state[types.int8(1)]}")
op_state(state,"op1")
print(f"add {state[types.int8(1)]}") 
op_state(state,"op2")
print(f"sub {state[types.int8(1)]}")             



