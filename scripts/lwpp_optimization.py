import tensorcircuit as tc
import numpy as np
import tensorflow as tf

K = tc.set_backend("tensorflow")
tc.set_dtype("float64")
tf.keras.backend.set_floatx("float64")

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

multiply_rule = {
    ("X", "Y"): ("Z", 1),
    ("Y", "X"): ("Z", -1),
    ("Y", "Z"): ("X", 1),
    ("Z", "Y"): ("X", -1),
    ("Z", "X"): ("Y", 1),
    ("X", "Z"): ("Y", -1),
    ("X", "X"): ("I", 1),
    ("Y", "Y"): ("I", 1),
    ("Z", "Z"): ("I", 1),
    ("I", "I"): ("I", 1),
    ("I", "X"): ("X", 1),
    ("X", "I"): ("X", 1),
    ("I", "Y"): ("Y", 1),
    ("Y", "I"): ("Y", 1),
    ("I", "Z"): ("Z", 1),
    ("Z", "I"): ("Z", 1),
}


def vary_pw(pw, ps_dict):
    pw_list = list(pw)
    for idx, po in ps_dict.items():
        pw_list[idx] = po
    return "".join(pw_list)


def weight_cal(pw):
    return pw.count("X") + pw.count("Y") + pw.count("Z")


class PauliSentence:
    def __init__(self, ps_dict={}):
        self.ps_dict = ps_dict

    # add PauliSentence
    def append_ps(self, ps_dict={}):
        for pw, coeff in ps_dict.items():
            if pw in self.ps_dict:
                self.ps_dict[pw] += coeff
            else:
                self.ps_dict[pw] = coeff

    def apply_cnot(self, wires):
        append_ps = {}
        for pw, coeff in self.ps_dict.items():
            if pw[wires[0]] != "I" or pw[wires[1]] != "I":
                new_pw_wires, factor = cnot_table[(pw[wires[0]], pw[wires[1]])]
                new_pw = vary_pw(
                    pw, {wires[0]: new_pw_wires[0], wires[1]: new_pw_wires[1]}
                )
                if weight_cal(new_pw) <= k:
                    append_ps[new_pw] = coeff * factor
            else:
                append_ps[pw] = coeff
        self.ps_dict = append_ps

    def apply_x(self, wires):
        for pw, coeff in self.ps_dict.items():
            if pw[wires] == "Z":
                self.ps_dict[pw] = -coeff
            elif pw[wires] == "Y":
                self.ps_dict[pw] = -coeff

    def apply_h(self, wires):
        append_ps = {}
        for pw, coeff in self.ps_dict.items():
            if pw[wires] == "I":
                append_ps[pw] = coeff
            elif pw[wires] == "X":
                new_pw = vary_pw(pw, {wires: "Z"})
                append_ps[new_pw] = coeff
            elif pw[wires] == "Z":
                new_pw = vary_pw(pw, {wires: "X"})
                append_ps[new_pw] = coeff
            elif pw[wires] == "Y":
                append_ps[pw] = -coeff
        self.ps_dict = append_ps

    def apply_single_rotation(self, wires, po, param):
        append_ps = {}
        for pw, coeff in self.ps_dict.items():
            po0 = pw[wires]
            if po0 != "I" and po0 != po:
                self.ps_dict[pw] = coeff * tf.cos(param)
                new_po0, factor = multiply_rule[(po0, po)]
                new_pw = vary_pw(pw, {wires: new_po0})
                append_ps[new_pw] = coeff * tf.sin(param) * factor
        self.append_ps(append_ps)

    def apply_single_rotation_sp(self, wires, po, param_cos, param_sin):
        append_ps = {}
        for pw, coeff in self.ps_dict.items():
            po0 = pw[wires]
            if po0 != "I" and po0 != po:
                self.ps_dict[pw] = coeff * param_cos
                new_po0, factor = multiply_rule[(po0, po)]
                new_pw = vary_pw(pw, {wires: new_po0})
                append_ps[new_pw] = coeff * param_sin * factor
        self.append_ps(append_ps)

    # provide params
    def apply_double_rotation(self, wires, po, param):
        append_ps = {}
        for pw, coeff in self.ps_dict.items():
            po0 = pw[wires[0]]
            po1 = pw[wires[1]]
            if (po0 != "I" and po0 != po) ^ (po1 != "I" and po1 != po):
                self.ps_dict[pw] = coeff * tf.cos(param)
                new_po0, factor0 = multiply_rule[(po0, po)]
                new_po1, factor1 = multiply_rule[(po1, po)]
                new_pw = vary_pw(pw, {wires[0]: new_po0, wires[1]: new_po1})
                if weight_cal(new_pw) <= k:
                    append_ps[new_pw] = coeff * tf.sin(param) * factor0 * factor1
        self.append_ps(append_ps)

    # provide cos(params) and sin(params)
    def apply_double_rotation_sp(self, wires, po, param_cos, param_sin):
        append_ps = {}
        for pw, coeff in self.ps_dict.items():
            po0 = pw[wires[0]]
            po1 = pw[wires[1]]
            if (po0 != "I" and po0 != po) ^ (po1 != "I" and po1 != po):
                self.ps_dict[pw] = coeff * param_cos
                new_po0, factor0 = multiply_rule[(po0, po)]
                new_po1, factor1 = multiply_rule[(po1, po)]
                new_pw = vary_pw(pw, {wires[0]: new_po0, wires[1]: new_po1})
                if weight_cal(new_pw) <= k:
                    append_ps[new_pw] = coeff * param_sin * factor0 * factor1
        self.append_ps(append_ps)

    def initial_state_expval(self):
        expval = 0.0
        for pauli_word, coeff in self.ps_dict.items():
            if set(pauli_word).issubset({"I", "Z"}):
                expval += coeff
        return expval


def build_circuit_XX_2d(params):
    c = tc.Circuit(num_qubits)
    for i in range(1, num_qubits, 2):
        c.x(i - 1)
        c.h(i - 1)
        c.cx(i - 1, i)
        c.x(i)
    for k_layers in range(num_layers):
        for k_params in range(num_params):
            c.rxx(
                la_edges[k_params][0],
                la_edges[k_params][1],
                theta=params[k_layers, k_params, 0],
            )
        for k_params in range(num_params):
            c.ryy(
                la_edges[k_params][0],
                la_edges[k_params][1],
                theta=params[k_layers, k_params, 1],
            )
        for k_params in range(num_params):
            c.rzz(
                la_edges[k_params][0],
                la_edges[k_params][1],
                theta=params[k_layers, k_params, 2],
            )
    return c


def build_Heisenberg_2d(Jx, Jy, Jz, hz):
    H = {}
    for k_params in range(len(la_edges)):
        pw_x = vary_pw(
            "I" * num_qubits, {la_edges[k_params][0]: "X", la_edges[k_params][1]: "X"}
        )
        pw_y = vary_pw(
            "I" * num_qubits, {la_edges[k_params][0]: "Y", la_edges[k_params][1]: "Y"}
        )
        pw_z = vary_pw(
            "I" * num_qubits, {la_edges[k_params][0]: "Z", la_edges[k_params][1]: "Z"}
        )
        H[pw_x] = Jx
        H[pw_y] = Jy
        H[pw_z] = Jz
    return PauliSentence(H)


def execute_tc(params):
    expval = 0
    c = build_circuit_XX_2d(params)
    ps = build_Heisenberg_2d(Jx, Jy, Jz, hz)
    for pw, coeffs in ps.ps_dict.items():
        x_pw = [i for i, char in enumerate(pw) if char == "X"]
        y_pw = [i for i, char in enumerate(pw) if char == "Y"]
        z_pw = [i for i, char in enumerate(pw) if char == "Z"]
        expval += coeffs * K.real(c.expectation_ps(x=x_pw, y=y_pw, z=z_pw))
    return K.real(expval)


def execute_ppa_sp_2d(params):
    params_cos = tf.cos(params)
    params_sin = tf.sin(params)
    ps = build_Heisenberg_2d(Jx, Jy, Jz, hz)

    for k_layers in range(num_layers)[::-1]:
        thetas_cos = params_cos[k_layers]
        thetas_sin = params_sin[k_layers]
        for k_params in range(num_params)[::-1]:
            ps.apply_double_rotation_sp(
                (la_edges[k_params][0], la_edges[k_params][1]),
                "Z",
                thetas_cos[k_params, 2],
                thetas_sin[k_params, 2],
            )
        for k_params in range(num_params)[::-1]:
            ps.apply_double_rotation_sp(
                (la_edges[k_params][0], la_edges[k_params][1]),
                "Y",
                thetas_cos[k_params, 1],
                thetas_sin[k_params, 1],
            )
        for k_params in range(num_params)[::-1]:
            ps.apply_double_rotation_sp(
                (la_edges[k_params][0], la_edges[k_params][1]),
                "X",
                thetas_cos[k_params, 0],
                thetas_sin[k_params, 0],
            )
    for i in range(1, num_qubits, 2)[::-1]:
        ps.apply_x(i)
        ps.apply_cnot((i - 1, i))
        ps.apply_h(i - 1)
        ps.apply_x(i - 1)

    result = ps.initial_state_expval()
    return result


# define lattice and circuit depth for scan
Lx, Ly = 2, 2
scan_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# k for LWPP initialization
scan_k = [2, 3]

iterations = 1500
ncircuits = 30


# Model type: 'm' for ferromagnetic, 'p' for antiferromagnetic
# mp = 'p'
mp = "m"

# Initialization: 'nz' for near-zero (identity), 'rd' for random
ini = "nz"
# ini = 'rd'



if mp == "m":
    Jx, Jy, Jz, hz = -1, -0.8, -0.5, -0.00
elif mp == "p":
    Jx, Jy, Jz, hz = 1, 0.8, 0.5, 0.00

scan_loss = np.zeros((len(scan_layers), ncircuits, iterations))

num_qubits = Lx * Ly
lattice = tc.templates.graphs.Grid2DCoord(Lx, Ly).lattice_graph(pbc=True)
la_edges = list(lattice.edges)
params_per_layer = 3
num_params = len(la_edges)

for k_i in range(len(scan_k)):
    for layers_i in range(len(scan_layers)):
        k = scan_k[k_i]
        num_layers = scan_layers[layers_i]
        print(f"k = {k}, layers = {num_layers}")

        execute_ppa_vvag_jit = K.jit(
            tc.backend.vvag(execute_ppa_sp_2d, vectorized_argnums=0)
        )
        np.random.seed(800)
        if ini == "nz":
            paramsv = (
                np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(ncircuits, num_layers, num_params, params_per_layer),
                )
                * np.pi
                * 0.01
            )
        elif ini == "rd":
            paramsv = (
                np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(ncircuits, num_layers, num_params, params_per_layer),
                )
                * np.pi
            )

        opt = K.optimizer(tf.keras.optimizers.Adam(1e-2))

        list_of_loss = [[] for i in range(ncircuits)]

        for i in range(iterations):
            loss, grads = execute_ppa_vvag_jit(paramsv)
            paramsv = opt.update(grads, paramsv)  # gradient descent

        np.save(
            f"../data/paramsv_set/paramsv_{mp}_{ini}_k{k}_nc{ncircuits}_l{num_layers}_n{Lx}{Ly}.npy",
            paramsv,
        )







