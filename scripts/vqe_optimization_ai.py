import time
from pathlib import Path

import jax
import numpy as np
import optax
import tensorcircuit as tc

# --- 1. Global Setup ---
print("JAX devices:", jax.devices())
tc.set_dtype("complex128")
K = tc.set_backend("jax")
tc.set_contractor("cotengra-8192-8192")


def vqe_forward(params, circuit_template):
    """
    Constructs the circuit and calculates the expectation value.
    This function is designed to be JIT-compiled.
    """
    # circuit_template already has num_qubits and edges baked in
    c = circuit_template(params)
    # The Hamiltonian 'h' is a global variable
    loss = tc.templates.measurements.operator_expectation(c, h)
    return loss


@K.jit
def train_step(param, opt_state, vvg_func, optimizer):
    """
    Performs a single, JIT-compiled training step.
    """
    loss_val, grads = vvg_func(param)
    updates, opt_state = optimizer.update(grads, opt_state, param)
    param = optax.apply_updates(param, updates)
    return param, opt_state, loss_val


# --- 2. Main Script Logic ---
if __name__ == "__main__":
    # --- Simulation Parameters ---
    # Lattice and circuit depth for scan
    Lx, Ly = 2, 2
    scan_layers = [2, 3]

    # k for LWPP initialization (k=0 means direct initialization)
    scan_k = [0, 2, 3]

    # Number of circuits for the LWPP-initialized parameters
    nc0 = 30
    ncircuits = 30  # Number of circuits to run in parallel
    iterations = 1500
    learning_rate = 1e-2

    # 'p': antiferromagnetic, 'm': ferromagnetic
    model_type = "p"
    # 'rd': random, 'nz': near-zero (near identity)
    init_type = "rd"

    # --- Setup Hamiltonian and Paths ---
    num_qubits = Lx * Ly

    # Define model parameters
    if model_type == "m":
        Jx, Jy, Jz = -1.0, -0.8, -0.5
    else:  # 'p'
        Jx, Jy, Jz = 1.0, 0.8, 0.5

    # Prepare directories for saving results
    Path("./loss_set").mkdir(exist_ok=True)
    Path("./paramsv_set").mkdir(exist_ok=True)

    # Setup lattice and Hamiltonian (as a global variable)
    coord = tc.templates.graphs.Grid2DCoord(Lx, Ly)
    lattice = coord.lattice_graph(pbc=True)
    edges = list(lattice.edges)
    num_params = len(edges)
    h = tc.quantum.heisenberg_hamiltonian(lattice, hxx=Jx, hyy=Jy, hzz=Jz, sparse=True)

    # --- Main Loops ---
    for num_layers in scan_layers:
        # Define a circuit template function for the current number of layers
        def circuit_template(params):
            c = tc.Circuit(num_qubits)
            # Initial layer
            for i in range(1, num_qubits, 2):
                c.x(i - 1)
                c.h(i - 1)
                c.cx(i - 1, i)
                c.x(i)
            # Variational layers
            for k_layer in range(num_layers):
                for k_param in range(num_params):
                    n1, n2 = edges[k_param]
                    c.rxx(n1, n2, theta=params[k_layer, k_param, 0])
                    c.ryy(n1, n2, theta=params[k_layer, k_param, 1])
                    c.rzz(n1, n2, theta=params[k_layer, k_param, 2])
            return c

        # Create the value-and-grad function for this specific circuit structure
        vvgf = K.jit(
            K.vectorized_value_and_grad(vqe_forward, argnums=0),
            static_argnums=(1),  # JIT-compile based on the circuit template
        )

        for k in scan_k:
            start_time = time.time()
            print(f"--- Starting run: layers={num_layers}, k={k} ---")

            # Initialize parameters
            if k == 0:
                if init_type == "rd":
                    np.random.seed(800)
                    param = (
                        np.random.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(ncircuits, num_layers, num_params, 3),
                        )
                        * np.pi
                    )
                else:  # 'nz'
                    np.random.seed(800)
                    param = (
                        np.random.uniform(
                            low=-1.0,
                            high=1.0,
                            size=(ncircuits, num_layers, num_params, 3),
                        )
                        * np.pi
                        * 0.01
                    )
            else:
                param_path = f"./paramsv_set/paramsv_{model_type}_{init_type}_k{k}_nc{nc0}_l{num_layers}_n{Lx}{Ly}.npy"
                print(f"Loading parameters from: {param_path}")
                try:
                    param = np.load(param_path)
                    param = param[0:ncircuits]
                except FileNotFoundError:
                    print(f"Warning: File not found, skipping run: {param_path}\n")
                    continue

            # Setup optimizer
            optimizer = optax.adam(learning_rate=learning_rate)
            opt_state = optimizer.init(param)

            # Training loop
            loss_history = []
            for i in range(iterations):
                param, opt_state, loss = train_step(
                    param, opt_state, vvgf, optimizer, circuit_template=circuit_template
                )
                loss_history.append(K.numpy(loss))
                if (i + 1) % 500 == 0:
                    print(f"  Iteration {i+1}/{iterations}, Avg Loss: {np.mean(loss):.6f}")

            # Save results
            final_loss_history = np.array(loss_history).T
            if k == 0:
                loss_path = f"./loss_set/loss_{model_type}_{init_type}_nc{ncircuits}_l{num_layers}_n{Lx}{Ly}.npy"
                param_path = f"./paramsv_set/paramsv_{model_type}_{init_type}_nc{ncircuits}_l{num_layers}_n{Lx}{Ly}.npy"
                np.save(param_path, K.numpy(param))
                print(f"Parameters saved to: {param_path}")
            else:
                loss_path = f"./loss_set/loss_{model_type}_{init_type}_k{k}_nc{ncircuits}_l{num_layers}_n{Lx}{Ly}.npy"

            np.save(loss_path, final_loss_history)
            print(f"Loss history saved to: {loss_path}")
            end_time = time.time()
            print(f"--- Finished run in {end_time - start_time:.2f} seconds ---\n")




