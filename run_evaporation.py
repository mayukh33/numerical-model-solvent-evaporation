"""Run the two-component evaporation model and write the profiles.

Usage:  python3 run_evaporation.py
"""

import os

import numpy as np

from solvent_evaporation.components import ComponentLibrary
from solvent_evaporation.mesh import ScaledMesh
from solvent_evaporation.solver import EvaporationSolver
from solvent_evaporation.thermodynamics import Mixture

components = {
    "methanol": {
        "M": 32.04,           # molar mass, g/mol
        "rho_pure": 787.2,    # pure liquid density, kg/m^3
        "p_sat": 16981.0,     # saturation vapour pressure, Pa
        "Dg": 1.51e-5,        # gas-phase diffusivity, m^2/s
    },
    "1-butanol": {
        "M": 74.12,
        "rho_pure": 805.9,
        "p_sat": 887.0,
        "Dg": 0.81e-5
    },
}

# Liquid mutual diffusivities at infinite dilution, m^2/s.  Both directions.
diffusivities = {
    ("methanol", "1-butanol"): 0.587e-9,
    ("1-butanol", "methanol"): 1.838e-9,
}

library = ComponentLibrary(components, diffusivities)

# --- components ----------------------------------------------------------
# ORDER MATTERS: the A-in-B diffusivity sets the problem's time scale.
# p_sat=0.0 makes B non-volatile; drop it to let butanol evaporate.
component_A = library.component("methanol")
component_B = library.component("1-butanol")

# Filename tags, so downstream analysis scripts keep finding the same files.
tag_A, tag_B = "m", "b"

# --- ambient conditions and model constants ------------------------------
R = 8314.0        # gas constant, J/(kmol.K) -- pairs with M in g/mol above
T = 298.15        # temperature, K
rho_gas = 1.24    # ambient gas density, kg/m^3
scaled_b = 2.0    # outer gas boundary in eta; the gas layer is [1, SCALED_B]

# --- case ----------------------------------------------------------------
wt_frac_A = 0.5
dx = 0.01
dt = 5e-6
final_time = 0.6
store_period = 1000

output_dir = (f"../../profile_time_fipy_binary_"
              f"{wt_frac_A:.2f}_{dt:.2e}_{dx:.2e}")

# --- build ---------------------------------------------------------------
mixture = Mixture(component_A, component_B, wt_frac_A,
                  R=R, T=T, rho_gas=rho_gas, library=library)
mesh = ScaledMesh(dx, scaled_b)

# --- initial profiles ----------------------------------------------------
# Scalar, array of length mesh.nx, or a 2-tuple (for_A, for_B).
# Quasi-steady gas start:  (mesh.scaled_b - mesh.gas_x) / (mesh.scaled_b - 1.0)
liquid_ini = 1.0
gas_ini = 0.0

solver = EvaporationSolver(mixture, mesh,
                           liquid_ini=liquid_ini, gas_ini=gas_ini)

print(f"{component_A.name} (A) + {component_B.name} (B) | "
      f"wt_A = {wt_frac_A} | rhol_0 = {mixture.rhol_0:.2f} kg/m3")
print(f"gamma_A = {mixture.gamma_A:.4e} | gamma_B = {mixture.gamma_B:.4e} |")

# --- run and store -------------------------------------------------------
# The solver yields every step; store_period decides what is kept.
nx = mesh.nx
num_steps = np.ceil((final_time / dt) / store_period).astype(np.int64)

scaled_rhoml_arr = np.zeros((nx, num_steps + 1))
scaled_rhobl_arr = np.zeros((nx, num_steps + 1))
scaled_rhomg_arr = np.zeros((nx, num_steps + 1))
scaled_rhobg_arr = np.zeros((nx, num_steps + 1))
scaled_rhol_arr = np.zeros((nx, num_steps + 1))

Delta = []
Time = []
count = 1

print(f"Final time: {final_time:.4f} | dt: {dt:.2e} | dx: {dx:.2e} | "
      f"nx: {nx:.1e} | store every {store_period} steps")

# t = 0 column.  Assigning into the arrays copies, so State's live views are safe.
state = solver.initial_state()
Delta.append(state.delta)
Time.append(state.t)
scaled_rhoml_arr[:, 0] = state.rho_l_A
scaled_rhobl_arr[:, 0] = state.rho_l_B
scaled_rhomg_arr[:, 0] = state.rho_g_A
scaled_rhobg_arr[:, 0] = state.rho_g_B
scaled_rhol_arr[:, 0] = state.rho_l

try:
    for state in solver.run(final_time, dt):
        if state.step % store_period == 0:
            print(f"Time: {state.t:.4f} | Delta: {state.delta:.6f} | "
                  f"dDelta/dt: {state.ddelta_dt:.6e}| "
                  f"rhol:{state.rhol[-1]:.2f}| "
                  f"scaled_rhol:{state.rho_l[-1]:.4f}|Error:{state.err}")
            Delta.append(state.delta)
            Time.append(state.t)
            scaled_rhoml_arr[:, count] = state.rho_l_A
            scaled_rhobl_arr[:, count] = state.rho_l_B
            scaled_rhomg_arr[:, count] = state.rho_g_A
            scaled_rhobg_arr[:, count] = state.rho_g_B
            scaled_rhol_arr[:, count] = state.rho_l
            count += 1
except KeyboardInterrupt:
    print(f"\nInterrupted at t = {state.t:.4f} (step {state.step}): "
          f"saving the {count} profiles stored so far.")


os.makedirs(output_dir, exist_ok=True)

np.savetxt(f"{output_dir}/scaled_rho{tag_A}l_fipy.dat",
           np.column_stack((mesh.liquid_x, scaled_rhoml_arr[:, :count])))
np.savetxt(f"{output_dir}/scaled_rho{tag_B}l_fipy.dat",
           np.column_stack((mesh.liquid_x, scaled_rhobl_arr[:, :count])))
np.savetxt(f"{output_dir}/scaled_rhol_fipy.dat",
           np.column_stack((mesh.liquid_x, scaled_rhol_arr[:, :count])))
np.savetxt(f"{output_dir}/scaled_rho{tag_A}g_fipy.dat",
           np.column_stack((mesh.gas_x, scaled_rhomg_arr[:, :count])))
np.savetxt(f"{output_dir}/scaled_rho{tag_B}g_fipy.dat",
           np.column_stack((mesh.gas_x, scaled_rhobg_arr[:, :count])))
np.savetxt(f"{output_dir}/delta_time_fipy.dat",
           np.column_stack((np.array(Time), np.array(Delta))))

print(f"Wrote {count} profile columns to {output_dir}")
