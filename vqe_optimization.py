from jax import config
import time
import optax
import tensorcircuit as tc
import random
import numpy as np
import jax

print(jax.devices())

tc.set_dtype("complex128")
K = tc.set_backend("jax")
tc.set_contractor("cotengra-8192-8192")


Lx, Ly = 3,3
num_qubits = Lx*Ly
num_layers = 1
Jx, Jy, Jz, hz = 1, 0.8, 0.5, 0
ncircuits = 30
nc0 = 30
iterations = 1500
k = 3


lattice = tc.templates.graphs.Grid2DCoord(Lx,Ly).lattice_graph(pbc = True)
la_edges = list(lattice.edges)
random.seed(10)
random.shuffle(la_edges)

num_params = len(la_edges)
params_per_layer = 3



coord = tc.templates.graphs.Grid2DCoord(Lx, Ly)
h = tc.quantum.heisenberg_hamiltonian(
    coord.lattice_graph(), hxx=Jx, hyy=Jy, hzz=Jz, sparse=True
)

def vary_pw(pw, pauli_dict):
    pw_list = list(pw)
    for idx, po in pauli_dict.items():
        pw_list[idx] = po
    return ''.join(pw_list)

def vqe_forward(params):
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
    loss = tc.templates.measurements.operator_expectation(c, h)
    return loss

np.random.seed(800)
params = np.random.random((num_layers, num_params, params_per_layer))*2*np.pi
val_exact = vqe_forward(params)

start1 = time.time()
val_exact = vqe_forward(params)
end1 = time.time()

time1 = end1-start1
print(f"Exact expectation value:      {val_exact:.6f}    Run time: {time1:.1f}s")


# @K.jit
# def train_step(param, opt_state):
#     # always using jitted optax paradigm when running on GPU!
#     loss_val, grads = vvgf(param)
#     updates, opt_state = optimizer.update(grads, opt_state, param)
#     param = optax.apply_updates(param, updates)
#     return param, opt_state, loss_val


# scan_layers = [2,3,4,5,6,7,8,9,10]
# scan_loss = np.zeros((len(scan_layers), ncircuits, iterations))

# scan_k = [0,2,3]


# mp = 'p'
# # mp = 'm'

# # ini = 'nz'
# ini = 'rd'


# if mp == 'm':
#     Jx, Jy, Jz, hz = -1,-0.8,-0.5,-0.00
# elif mp == 'p':
#     Jx, Jy, Jz, hz = 1,0.8,0.5,0.00


# coord = tc.templates.graphs.Grid2DCoord(Lx, Ly)
# h = tc.quantum.heisenberg_hamiltonian(
#     coord.lattice_graph(), hxx=Jx, hyy=Jy, hzz=Jz, sparse=True
# )

# scan_loss = np.zeros((len(scan_k),len(scan_layers),ncircuits,iterations))

# for layers_i in range(len(scan_layers)):
#     for k_i in range(len(scan_k)):
#         num_layers = scan_layers[layers_i]
#         k = scan_k[k_i]
#         vvgf = K.jit(
#             K.vectorized_value_and_grad(vqe_forward),
#         )
#         if k == 0:
#             if ini == 'rd':
#                 np.random.seed(800)
#                 param = np.random.uniform(low = -1.0, high = 1.0,
#                     size = (ncircuits,num_layers,num_params,params_per_layer))*np.pi
#             else:
#                 np.random.seed(800)
#                 param = np.random.uniform(low = -1.0, high = 1.0,
#                     size = (ncircuits,num_layers,num_params,params_per_layer))*np.pi*0.01
#         else:
#             param = np.load(f'./paramsv_set/paramsv_{mp}_{ini}_k{k}_nc{nc0}_l{num_layers}_n{Lx}{Ly}.npy')
#             param = param[0:ncircuits]
#         # param = jnp.array(param)        
#         list_of_loss = [[] for i in range(ncircuits)]
        
#         optimizer = optax.adam(learning_rate=1e-2)
#         opt_state = optimizer.init(param)
                                
        
#         for i in range(iterations):
#             param, opt_state, loss = train_step(param, opt_state)
#             list_of_loss = np.hstack((list_of_loss, K.numpy(loss)[:, np.newaxis]))
#             if np.mod(i,100) == 0:
#                 print(i)
#         scan_loss[k_i,layers_i,:] = list_of_loss
#         if k == 0:
#             np.save(f'./loss_set/loss_{mp}_{ini}_nc{ncircuits}_l{num_layers}_n{Lx}{Ly}',list_of_loss)
#         else:
#             np.save(f'./loss_set/loss_{mp}_{ini}_k{k}_nc{ncircuits}_l{num_layers}_n{Lx}{Ly}.npy',list_of_loss)
#         print(f'k = {k}     num_layers = {num_layers}')
        
# np.save('./loss_set/scan_loss_{mp}_{ini}_nc{ncircuits}_n{Lx}{Ly}.npy',scan_loss)









