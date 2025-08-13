import tensorcircuit as tc
import numpy as np
import time
import random
import tensorflow as tf
import scipy

K = tc.set_backend('tensorflow')
tc.set_dtype('float64')
tf.keras.backend.set_floatx('float64')


cnot_table = {
    ("I", "I"): (("I", "I"), 1),
    ("I", "X"): (("I", "X"), 1),
    ("I", "Y"): (("Z", "Y"), 1),
    ("I", "Z"): (("Z", "Z"), 1),
    ("X", "I"): (("X", "X"), 1),
    ("X", "X"): (("X", "I"), 1),
    ("X", "Y"): (("Y", "Z"), 1),
    ("X", "Z"): (("Y", "Y"), -1),
    ("Y", "I"): (("Y", "X"), 1),
    ("Y", "X"): (("Y", "I"), 1),
    ("Y", "Y"): (("X", "Z"), -1),
    ("Y", "Z"): (("X", "Y"), 1),
    ("Z", "I"): (("Z", "I"), 1),
    ("Z", "X"): (("Z", "X"), 1),
    ("Z", "Y"): (("I", "Y"), 1),
    ("Z", "Z"): (("I", "Z"), 1),
}

multiply_rule = {('X','Y'):('Z',1), ('Y','X'):('Z',-1),
                    ('Y','Z'):('X',1), ('Z','Y'):('X',-1),
                    ('Z','X'):('Y',1), ('X','Z'):('Y',-1),
                    ('X','X'):('I',1), ('Y','Y'):('I',1),
                    ('Z','Z'):('I',1), ('I','I'):('I',1),
                    ('I','X'):('X',1), ('X','I'):('X',1),
                    ('I','Y'):('Y',1), ('Y','I'):('Y',1),
                    ('I','Z'):('Z',1), ('Z','I'):('Z',1)
                }  

def vary_pw(pw, dict):
    pw_list = list(pw)
    for idx, po in dict.items():
        pw_list[idx] = po
    return ''.join(pw_list)

def pw2array(pw):
    result = [0 for i in range(len(pw))];
    for k in range(len(pw)):
        if pw[k] == 'I':
            result[k] = 0
        elif pw[k] == 'X':
            result[k] = 1
        elif pw[k] == 'Y':
            result[k] = 2
        elif pw[k] == 'Z':
            result[k] = 3
    return result

def weight_cal(pw):
    return pw.count('X')+pw.count('Y')+pw.count('Z')


class PauliSentence:
    def __init__(self, dict = {}):
        self.dict = dict

    # add PauliSentence
    def append_ps(self, dict = {}):
        for pw, coeff in dict.items():
            if pw in self.dict:
                self.dict[pw] += coeff
            else:
                self.dict[pw] = coeff
            
    def apply_cnot(self, wires):
        append_ps = {}
        for pw, coeff in self.dict.items():
            if pw[wires[0]] != 'I' or pw[wires[1]] != 'I':
                new_pw_wires, factor = cnot_table[(pw[wires[0]], pw[wires[1]])]
                new_pw = vary_pw(pw, {wires[0]: new_pw_wires[0], wires[1]: new_pw_wires[1]})
                if weight_cal(new_pw) <= k:
                    append_ps[new_pw] = coeff*factor
            else:
                append_ps[pw] = coeff
        self.dict = append_ps
    
    def apply_x(self, wires):
        for pw, coeff in self.dict.items():
            if pw[wires] == 'Z':
                self.dict[pw] = -coeff
            elif pw[wires] == 'Y':
                self.dict[pw] = -coeff
    
    def apply_h(self, wires):
        append_ps = {}
        for pw, coeff in self.dict.items():
            if pw[wires] == 'I':
                append_ps[pw] = coeff
            elif pw[wires] == 'X':
                new_pw = vary_pw(pw, {wires:'Z'})
                append_ps[new_pw] = coeff
            elif pw[wires] == 'Z':
                new_pw = vary_pw(pw, {wires:'X'})
                append_ps[new_pw] = coeff
            elif pw[wires] == 'Y':
                append_ps[pw] = -coeff
        self.dict = append_ps
    
    def apply_single_rotation(self, wires, po, param):
        append_ps = {}
        for pw, coeff in self.dict.items():
            po0 = pw[wires]
            if po0 != 'I' and po0 != po:
                self.dict[pw] = coeff*tf.cos(param)
                new_po0, factor = multiply_rule[(po0, po)]
                new_pw = vary_pw(pw, {wires:new_po0})
                append_ps[new_pw] = coeff*tf.sin(param)*factor
        self.append_ps(append_ps)
        
    def apply_single_rotation_sp(self, wires, po, param_cos, param_sin):
        append_ps = {}
        for pw, coeff in self.dict.items():
            po0 = pw[wires]
            if po0 != 'I' and po0 != po:
                self.dict[pw] = coeff*param_cos
                new_po0, factor = multiply_rule[(po0, po)]
                new_pw = vary_pw(pw, {wires:new_po0})
                append_ps[new_pw] = coeff*param_sin*factor
        self.append_ps(append_ps)
        
    # provide params
    def apply_double_rotation(self, wires, po, param):
        append_ps = {}
        for pw, coeff in self.dict.items():
            po0 = pw[wires[0]]
            po1 = pw[wires[1]]
            if (po0 != 'I' and po0 != po) ^ (po1 != 'I' and po1 != po):
                self.dict[pw] = coeff*tf.cos(param)
                new_po0, factor0 = multiply_rule[(po0, po)]
                new_po1, factor1 = multiply_rule[(po1, po)]
                new_pw = vary_pw(pw, {wires[0]:new_po0, wires[1]:new_po1})
                if weight_cal(new_pw) <= k:
                    append_ps[new_pw] = coeff*tf.sin(param)*factor0*factor1
        self.append_ps(append_ps)
        
    # provide cos(params) and sin(params)
    def apply_double_rotation_sp(self, wires, po, param_cos, param_sin):
        append_ps = {}
        for pw, coeff in self.dict.items():
            po0 = pw[wires[0]]
            po1 = pw[wires[1]]
            if (po0 != 'I' and po0 != po) ^ (po1 != 'I' and po1 != po):
                self.dict[pw] = coeff*param_cos
                new_po0, factor0 = multiply_rule[(po0, po)]
                new_po1, factor1 = multiply_rule[(po1, po)]
                new_pw = vary_pw(pw, {wires[0]:new_po0, wires[1]:new_po1})
                if weight_cal(new_pw) <= k:
                    append_ps[new_pw] = coeff*param_sin*factor0*factor1
        self.append_ps(append_ps)

                
    def initial_state_expval(self):
        expval = 0.0
        for pauli_word, coeff in self.dict.items():
            if set(pauli_word).issubset({'I', 'Z'}):
                expval += coeff
        return expval




def build_circuit_XX_2d(params):
    c = tc.Circuit(num_qubits)
    for i in range(1, num_qubits, 2):
        c.x(i-1)
        c.h(i-1)
        c.cx(i-1, i)
        c.x(i)
    for k_layers in range(num_layers):
        for k_params in range(num_params):
            c.rxx(la_edges[k_params][0], la_edges[k_params][1], theta = params[k_layers, k_params, 0])
        for k_params in range(num_params):
            c.ryy(la_edges[k_params][0], la_edges[k_params][1], theta = params[k_layers, k_params, 1])
        for k_params in range(num_params):
            c.rzz(la_edges[k_params][0], la_edges[k_params][1], theta = params[k_layers, k_params, 2])
    return c


def build_Heisenberg_2d(Jx,Jy,Jz,hz):
    H = {}
    for k_params in range(len(la_edges)):
        pw_x = vary_pw('I'*num_qubits,{la_edges[k_params][0]:'X',la_edges[k_params][1]:'X'})
        pw_y = vary_pw('I'*num_qubits,{la_edges[k_params][0]:'Y',la_edges[k_params][1]:'Y'})
        pw_z = vary_pw('I'*num_qubits,{la_edges[k_params][0]:'Z',la_edges[k_params][1]:'Z'})
        H[pw_x] = Jx
        H[pw_y] = Jy
        H[pw_z] = Jz
    return PauliSentence(H)

# def build_H2O():

#     geometry = [['O', [0.0, 0.0, 0.0]], ['H', [0.757, 0.586, 0.0]], ['H', [-0.757, 0.586, 0.0]]]
#     basis = 'sto-3g'
#     multiplicity = 1
#     charge = 0


#     molecule = MolecularData(geometry, basis, multiplicity, charge, filename='h2o')


#     molecule = run_pyscf(molecule, run_scf=1, run_ccsd=1)


#     fermionic_hamiltonian = molecule.get_molecular_hamiltonian()
#     qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)
#     qubit_hamiltonian.compress()
    
#     H = {}
#     for term, coeff in list(qubit_hamiltonian.terms.items())[:72]:
#         pw = 'I'*num_qubits
#         for k in range(len(term)):
#             pw = vary_pw(pw,{term[k][0]:term[k][1]})
#         H[pw] = coeff

#     return PauliSentence(H)


def execute_tc(params):
    expval = 0
    # if ws and dr_pair == '1d' and apply_seq == 'XY':
    #     c = build_circuit_XY_1d_ws(params)
    # elif ws and dr_pair == '1d' and apply_seq == 'XX':
    #     c = build_circuit_XX_1d_ws(params)
    # elif ws and dr_pair == '2d' and apply_seq == 'XY':
    #     c = build_circuit_XY_2d_ws(params)
    # elif ws and dr_pair == '2d' and apply_seq == 'XX':
    #     c = build_circuit_XX_2d_ws(params)
    # elif ws == False and dr_pair == '1d' and apply_seq == 'XY':
    #     c = build_circuit_XY_1d(params)
    # elif ws == False and dr_pair == '1d' and apply_seq == 'XX':
    #     c = build_circuit_XX_1d(params)
    # elif ws == False and dr_pair == '2d' and apply_seq == 'XY':
    #     c = build_circuit_XY_2d(params)
    # elif ws == False and dr_pair == '2d' and apply_seq == 'XX':
    c = build_circuit_XX_2d(params)
        
    ps = build_Heisenberg_2d(Jx, Jy, Jz, hz)
    for pw, coeffs in ps.dict.items():
        x_pw = [i for i, char in enumerate(pw) if char == 'X']
        y_pw = [i for i, char in enumerate(pw) if char == 'Y']
        z_pw = [i for i, char in enumerate(pw) if char == 'Z']
        expval += coeffs*K.real(c.expectation_ps(x = x_pw, y = y_pw, z = z_pw))
    return K.real(expval)


# def ite_probe(list_of_loss, tmp_e0, acc):
#     tmp_0 = 0
#     tmp_n = 0
#     tmp_acc = acc
#     list_shape = list_of_loss.shape
#     for i in range(list_shape[1]):
#         tmp_loss = list_of_loss[:,i]
#         if tmp_0 == 0 and abs((np.min(tmp_loss)-tmp_e0)/tmp_e0) < tmp_acc:
#             tmp_000 = 1
#             return i
#     return 0




def execute_ppa_sp_2d(params):
    params_cos = tf.cos(params)
    params_sin = tf.sin(params)
    ps = build_Heisenberg_2d(Jx,Jy,Jz,hz)

    for k_layers in range(num_layers)[::-1]:
        thetas_cos = params_cos[k_layers]
        thetas_sin = params_sin[k_layers]
        for k_params in range(num_params)[::-1]:
#             ps.apply_double_rotation_sp((la_edges[k_params][0], la_edges[k_params][1]),'Z',
#                                         np.cos(np.pi*0.6),np.sin(np.pi*0.6)) 
            ps.apply_double_rotation_sp((la_edges[k_params][0], la_edges[k_params][1]),'Z',
                                        thetas_cos[k_params,2],thetas_sin[k_params,2])
        for k_params in range(num_params)[::-1]:
#             ps.apply_double_rotation_sp((la_edges[k_params][0], la_edges[k_params][1]),'Y',
#                                         np.cos(np.pi*0.8),np.sin(np.pi*0.8)) 
            ps.apply_double_rotation_sp((la_edges[k_params][0], la_edges[k_params][1]),'Y',
                                        thetas_cos[k_params,1],thetas_sin[k_params,1]) 
        for k_params in range(num_params)[::-1]:
#             ps.apply_double_rotation_sp((la_edges[k_params][0], la_edges[k_params][1]),'X',
#                                         np.cos(np.pi/3),np.sin(np.pi/3))  
            ps.apply_double_rotation_sp((la_edges[k_params][0], la_edges[k_params][1]),'X',
                                        thetas_cos[k_params,0],thetas_sin[k_params,0])    
    for i in range(1, num_qubits, 2)[::-1]:
        ps.apply_x(i)
        ps.apply_cnot((i-1,i))
        ps.apply_h(i-1)
        ps.apply_x(i-1)
    
#     print(len(ps.dict))
    result = ps.initial_state_expval()
    return result
    




Lx = 2
Ly = 2

num_qubits = Lx*Ly
num_layers = 2
k = 4

lattice = tc.templates.graphs.Grid2DCoord(Lx, Ly).lattice_graph(pbc=True)
la_edges = list(lattice.edges)
random.seed(10)
random.shuffle(la_edges)


# # Ansatz
dr_pair = '2d'
apply_seq = 'XX'
ws = False

if ws:
    params_per_layer = 5
else:
    params_per_layer = 3

if dr_pair == '1d':
    num_params = num_qubits
elif dr_pair == '2d':
    num_params = len(la_edges)
        

tf.random.set_seed(800)
params = tf.random.uniform((num_layers, num_params, params_per_layer))*2*np.pi




Jx, Jy, Jz, hz = 1,0.8,0.5,0.00
Jx, Jy, Jz, hz = -1,-0.8,-0.5,-0.00

h = tc.quantum.heisenberg_hamiltonian(lattice, hxx=Jx, hyy=Jy, hzz=Jz, hz = hz)
e0 = scipy.sparse.linalg.eigsh(K.numpy(h), k=1, which="SA")[0]
val_ppa = execute_ppa_sp_2d(params)


start1 = time.time()
val_exact = execute_tc(params)
end1 = time.time()

start0 = time.time()
val_ppa = execute_ppa_sp_2d(params)
end0 = time.time() 

time0 = end0-start0
time1 = end1-start1
print(f"Truncated Pauli propagation:  {val_ppa:.6f}    Run time: {time0:.1f}s")
print(f"Exact expectation value:      {val_exact:.6f}    Run time: {time1:.1f}s")







# scan_layers = [2,3,4,5,6,7,8,9,10]
# # scan_layers = [9,10]
# iterations = 1500
# ncircuits = 30

# mp = 'p'
# # mp = 'm'

# # ini = 'nz'
# ini = 'rd'


# if mp == 'm':
#     Jx, Jy, Jz, hz = -1,-0.8,-0.5,-0.00
# elif mp == 'p':
#     Jx, Jy, Jz, hz = 1,0.8,0.5,0.00

# scan_loss = np.zeros((len(scan_layers),ncircuits,iterations))
# scan_k = [2,3]



# lattice = tc.templates.graphs.Grid2DCoord(Lx, Ly).lattice_graph(pbc=True)
# la_edges = list(lattice.edges)
# random.seed(10)
# random.shuffle(la_edges)
# num_params = len(la_edges)

# for k_i in range(len(scan_k)):
#     for layers_i in range(len(scan_layers)):
#         k = scan_k[k_i]
#         num_layers = scan_layers[layers_i]
        
#         execute_ppa_vvag_jit = K.jit(tc.backend.vvag(execute_ppa_sp_2d, vectorized_argnums=0))
#         np.random.seed(800)
#         if ini == 'nz':
#             paramsv = np.random.uniform(low = -1.0, high = 1.0,
#                                        size = (ncircuits, num_layers, num_params, params_per_layer))*np.pi*0.01
#         elif ini == 'rd':
#             paramsv = np.random.uniform(low = -1.0, high = 1.0,
#                                        size = (ncircuits, num_layers, num_params, params_per_layer))*np.pi
    
    
#         opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))
    
#         list_of_loss = [[] for i in range(ncircuits)]
        
#         for i in range(iterations):
#             loss, grads = execute_ppa_vvag_jit(paramsv)
#             paramsv = opt.update(grads, paramsv)  # gradient descent
            
#         folder_path = './paramsv_set'
#         file_name = f'paramsv_{mp}_{ini}_k{k}_nc{ncircuits}_l{num_layers}_n{Lx}{Ly}.npy'
#         full_path = os.path.join(folder_path, file_name)
        
#         # 如果文件夹不存在，先创建
#         os.makedirs(folder_path, exist_ok=True)
            
#         np.save(full_path,paramsv)
        










