#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:20:08 2026
Modified for Gas-Phase Global Mass Balance
Enhanced with Predictor-Corrector iteration for non-linear stability.
"""

import numpy as np
import fipy as fp
import os

dx = 1e-2
dt = 1e-5
nx = int(np.floor(1/dx))

mesh_l = fp.Grid1D(nx=nx, dx=dx)
mesh_g = fp.Grid1D(nx=nx, dx=dx) + 1.0

etal = mesh_l.cellCenters.value[0]
etag = mesh_g.cellCenters.value[0]

wt_frac_M = 0.5 

print(f"dx: {dx}, dt: {dt}")
output_dir = f"profile_time_fipy_{wt_frac_M:.2f}_{dt:.2e}_{dx:.2e}-v2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
wt_frac_B = 1.0 - wt_frac_M

rho_M_pure = 787.2 
rho_B_pure = 805.9 
rhog_0 = 1.24

# Calculate mixture density assuming ideal solution (additive volumes)
rhol_0 = 1.0 / (wt_frac_M / rho_M_pure + wt_frac_B / rho_B_pure)
rhol = np.ones(nx) * rhol_0

print(f"rhol_0: {rhol_0}")
# Calculate initial mass concentration of methanol in the mixture
rhol_ini = wt_frac_M * rhol_0

# Physical parameters
Dmb0, Dbm0, Dg = 0.587e-9, 1.838e-9, 1.51e-5        
Mm, Mb = 32.04, 74.12       
pm0, R, T = 16981.0, 8314.0, 298.15          

x_M_ini = Mb * rhol_ini / (Mm * rhol_0 + Mb * rhol_ini - Mm * rhol_ini)
x_B_ini = 1.0 - x_M_ini
Ml_ini = Ml0 = (x_M_ini * Mm) + (x_B_ini * Mb)

alpha = Dmb0 / Dg
omega0 = rhol_ini / rhol_0
const_rho = rhol_0 / rhog_0            
gamma = 3.83e-4 # Ml0 * pm0 / (rhol_0 * R * T)
scaled_b = 2.0

delta = 1.0
ddelta_dt = 0

delta_var = fp.CellVariable(mesh=mesh_l, value=delta, hasOld=True)
Lg_var = fp.CellVariable(mesh=mesh_g, value=scaled_b - delta, hasOld=True)

uN_var = fp.Variable(value=1.0)
v0_var = fp.Variable(value=0.0)

scaled_rhoml = fp.CellVariable(mesh=mesh_l, value=1.0, hasOld=True)

scaled_Ml_ini = Ml_ini / Ml0
K_part_ini = 1.0 / scaled_Ml_ini
v0_initial = 1.0 / K_part_ini

scaled_rhomg = fp.CellVariable(mesh=mesh_g, value=0.0, hasOld=True)
scaled_rhoml.constrain(uN_var, mesh_l.facesRight)
scaled_rhomg.constrain(v0_var, mesh_g.facesLeft)
scaled_rhomg.constrain(0.0, mesh_g.facesRight)

eff_Dl_cv = fp.CellVariable(mesh=mesh_l, value=1.0)
Dg_eff_cv = fp.CellVariable(mesh=mesh_g, value=1.0)

v_liq_face = fp.FaceVariable(mesh=mesh_l, rank=1)
v_gas_face = fp.FaceVariable(mesh=mesh_g, rank=1)

eq_l = (fp.TransientTerm(coeff=delta_var) + 
        fp.UpwindConvectionTerm(coeff=v_liq_face) == 
        fp.DiffusionTerm(coeff=eff_Dl_cv.arithmeticFaceValue))

eq_g = (fp.TransientTerm(coeff=Lg_var) + 
        fp.UpwindConvectionTerm(coeff=v_gas_face) == 
        fp.DiffusionTerm(coeff=Dg_eff_cv.arithmeticFaceValue))

Delta = []
Time = []

# Starting from the second step since the first step is defined by t = 0 
# which is the initial condition 

step = 1
t_ = dt 
final_time = 1.4
count = 1

store_period = 1000
num_steps = np.int64((final_time/dt)/store_period)
scaled_rhoml_arr = np.zeros((nx, num_steps+1))
scaled_rhomg_arr = np.zeros((nx, num_steps+1))
scaled_rhol_arr = np.zeros((nx, num_steps+1))

# Initialize scaled gas mass integral before the loop
previous_Ig = np.float64(np.sum(scaled_rhomg.value) * dx)

Delta.append(1)
Time.append(0)

scaled_rhoml_arr[:, 0] = np.ones(100)
scaled_rhomg_arr[:, 0] = np.zeros(100)

# Settings for non-linear Predictor-Corrector iteration
max_inner_iters = 10
inner_tol = 1e-7

while t_ <= final_time: 
    err = 1
    # Store previous step states
    delta_old = delta
    ddelta_dt_old = ddelta_dt
    
    delta_var.updateOld()
    Lg_var.updateOld()
    scaled_rhoml.updateOld()
    scaled_rhomg.updateOld()
    
    # Predictor step initial guesses
    delta_pred = delta_old + ddelta_dt_old * dt
    ddelta_dt_actual = ddelta_dt_old
    
    # --- Predictor-Corrector Loop ---
    while err > inner_tol:
        delta_var.setValue(delta_pred)
        Lg_pred = scaled_b - delta_pred
        Lg_var.setValue(Lg_pred)
        
        # Evaluate local values dynamically during the iterations
        rhoml_val = scaled_rhoml.value
        rhomg_val = scaled_rhomg.value
        
        rhoml_arr = rhoml_val * rhol_ini
        rhol_val = rhoml_arr + rho_B_pure * (1.0 - (rhoml_arr / rho_M_pure))
        scaled_rhol_current = rhol_val / rhol_0
        
        x_M_arr = Mb * rhoml_arr / (Mm * rhol_val + Mb * rhoml_arr - Mm * rhoml_arr)
        x_B_arr = 1.0 - x_M_arr
         
        Ml_arr = (x_M_arr * Mm) + (x_B_arr * Mb)
        scaled_Ml_arr = Ml_arr / Ml0
        
        # Scaled Diffusivities
        Dl_arr = (Dmb0**x_B_arr) * (Dbm0**x_M_arr)
        eff_Dl_arr = (Dl_arr / Dmb0) / delta_pred
        Dg_eff_pred = (1.0 / alpha) / Lg_pred
        
        C_liq_bnd = 2.0 * eff_Dl_arr[-1] / dx 
        C_gas_bnd = 2.0 * Dg_eff_pred / dx
        K_part = scaled_rhol_current[-1] / scaled_Ml_arr[-1]
        
        numerator = C_liq_bnd * rhoml_val[-1] + gamma * C_gas_bnd * rhomg_val[0]
        denominator = gamma * C_gas_bnd + C_liq_bnd * K_part + ddelta_dt_actual * (K_part - gamma * const_rho * scaled_rhol_current[-1])

        v0 = numerator / denominator    
        uN = K_part * v0  
        
        # Setting new values for delta and diffusivities
        eff_Dl_cv.setValue(eff_Dl_arr)
        Dg_eff_cv.setValue(Dg_eff_pred)
        
        k = 1.0 - const_rho * scaled_rhol_current[-1]
        
        # Update face velocities
        v_liq_face.setValue(-mesh_l.faceCenters * ddelta_dt_actual)
        v_gas_face.setValue(ddelta_dt_actual * (mesh_g.faceCenters - 2.0 + k))
        
        # Update constraints
        uN_var.setValue(uN)
        v0_var.setValue(v0)
        
        # Solve PDEs 
        eq_l.solve(var=scaled_rhoml, dt=dt)
        eq_g.solve(var=scaled_rhomg, dt=dt)
        
        # Global mass balance & time derivatives 
        unscaled_rhoml_new = scaled_rhoml.value * rhol_ini
        rhol_new = unscaled_rhoml_new + rho_B_pure * (1.0 - (unscaled_rhoml_new / rho_M_pure))
        scaled_rhol_new = rhol_new / rhol_0
        scaled_rhol_new_int = scaled_rhol_new[-1]
        
        Ig_new = np.float64(np.sum(scaled_rhomg.value) * dx)
        dIg_dt = (Ig_new - previous_Ig) / dt
        
        grad_gas_right = (0.0 - scaled_rhomg.value[-1]) / (dx / 2.0)
        
        term1 = (omega0 * gamma / (alpha * Lg_pred)) * grad_gas_right
        term2 = omega0 * gamma * Lg_pred * dIg_dt
        denom_actual = scaled_rhol_new_int - (omega0 * gamma * Ig_new)
        
        ddelta_dt_actual = (term1 - term2) / denom_actual
        
        # Convergence check on interface position
        delta_next = delta_old + ddelta_dt_actual * dt
        err = abs(delta_next - delta_pred)
        if err < inner_tol:
            delta_pred = delta_next
            break
        
        delta_pred = delta_next
      
    # Update state variables for next time step
    ddelta_dt = ddelta_dt_actual
    previous_Ig = Ig_new
    rhol = rhol_new
    delta = delta_old + (ddelta_dt * dt)

    if step % store_period == 0:   
        print(f"Time: {t_:.4f} | Delta: {np.float64(delta):.6f} | dDelta/dt: {np.float64(ddelta_dt):.6e}| rhol:{np.float64(rhol[-1]):.2f}| scaled_rhol:{np.float64(scaled_rhol_new[-1]):.4f}|Error:{err}")
        Delta.append(delta)
        Time.append(t_)
        scaled_rhoml_arr[:, count] = scaled_rhoml.value
        scaled_rhomg_arr[:, count] = scaled_rhomg.value
        scaled_rhol_arr[:, count] = scaled_rhol_new
        count += 1 
    
    t_ += dt
    step += 1

np.savetxt(f"{output_dir}/scaled_rhoml_{wt_frac_M:.2f}_fipy.dat", np.column_stack((etal, scaled_rhoml_arr)))
np.savetxt(f"{output_dir}/scaled_rhol_{wt_frac_M:.2f}_fipy.dat", np.column_stack((etal, scaled_rhol_arr)))
np.savetxt(f"{output_dir}/scaled_rhomg_{wt_frac_M:.2f}_fipy.dat", np.column_stack((etag, scaled_rhomg_arr))) 
np.savetxt(f"{output_dir}/delta_time_fipy_{wt_frac_M:.2f}.dat", np.column_stack((np.array(Time), np.array(Delta))))
