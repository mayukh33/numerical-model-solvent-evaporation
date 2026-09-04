"""Bit-identical checks against the original flat script.

Runs under plain `python3 tests/test_port.py` -- no FiPy, no pytest.  Every
assertion demands an EXACT match (difference of 0.0), because the package is a
pure restructuring: any difference at all is a porting error.

The reference expressions below are transcribed verbatim from
solvent_evaporation-two-component.py so the two can be compared side by side.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvent_evaporation.components import Component, ComponentLibrary
from solvent_evaporation.global_flux import GlobalFluxBalance
from solvent_evaporation.local_flux import LocalFluxBalance
from solvent_evaporation.thermodynamics import Mixture

# --- constants exactly as the original script named them -------------------
wt_frac_M = 0.5
wt_frac_B = 1.0 - wt_frac_M
rho_M_pure, rho_B_pure, rhog_0 = 787.2, 805.9, 1.24
R, T = 8314.0, 298.15
SCALED_B = 2.0
Dmb0, Dbm0, Dg_M, Dg_B = 0.587e-9, 1.838e-9, 1.51e-5, 0.81e-5
Mm, Mb = 32.04, 74.12
pm0, pb0 = 16981.0, 0.0
dx = 0.01

rhol_0 = 1.0 / (wt_frac_M / rho_M_pure + wt_frac_B / rho_B_pure)
rhol_M_ini = wt_frac_M * rhol_0
rhol_B_ini = wt_frac_B * rhol_0
x_M_ini = (wt_frac_M / Mm) / (wt_frac_M / Mm + wt_frac_B / Mb)
x_B_ini = 1.0 - x_M_ini
Ml0 = (x_M_ini * Mm) + (x_B_ini * Mb)
alpha_M, alpha_B = Dmb0 / Dg_M, Dmb0 / Dg_B
omega0_M, omega0_B = rhol_M_ini / rhol_0, rhol_B_ini / rhol_0
const_rho = rhol_0 / rhog_0
gamma_M = pm0 * Ml0 / (rhol_0 * R * T)
gamma_B = pb0 * Ml0 / (rhol_0 * R * T)

# The package ships no substance data, so the test supplies its own -- the
# same numbers the original flat script hardcoded.
COMPONENTS = {
    "methanol": {"M": Mm, "rho_pure": rho_M_pure, "p_sat": pm0, "Dg": Dg_M},
    "1-butanol": {"M": Mb, "rho_pure": rho_B_pure, "p_sat": 887.0, "Dg": Dg_B},
}
DIFFUSIVITIES = {("methanol", "1-butanol"): Dmb0,
                 ("1-butanol", "methanol"): Dbm0}
LIB = ComponentLibrary(COMPONENTS, DIFFUSIVITIES)

METHANOL = LIB.component("methanol")
BUTANOL = LIB.component("1-butanol", p_sat=pb0)
mix = Mixture(METHANOL, BUTANOL, wt_frac_M,
              R=R, T=T, rho_gas=rhog_0, library=LIB)
failures = []


def check(label, expected, got):
    expected = np.asarray(expected, dtype=float)
    got = np.asarray(got, dtype=float)
    diff = float(np.max(np.abs(expected - got))) if expected.size else 0.0
    ok = diff == 0.0
    print(f"   {'PASS' if ok else 'FAIL'}  {label:<44} max|diff| = {diff:.3e}")
    if not ok:
        failures.append(label)


print("1. REFERENCE STATE")
for label, expected, got in [
        ("rhol_0", rhol_0, mix.rhol_0),
        ("rhol_M_ini", rhol_M_ini, mix.rhol_A_ini),
        ("rhol_B_ini", rhol_B_ini, mix.rhol_B_ini),
        ("x_M_ini", x_M_ini, mix.x_A_ini),
        ("Ml0", Ml0, mix.Ml0),
        ("const_rho", const_rho, mix.const_rho),
        ("omega0_M", omega0_M, mix.omega0_A),
        ("omega0_B", omega0_B, mix.omega0_B),
        ("gamma_M", gamma_M, mix.gamma_A),
        ("gamma_B", gamma_B, mix.gamma_B),
        ("alpha_M", alpha_M, mix.alpha_A),
        ("alpha_B", alpha_B, mix.alpha_B)]:
    check(label, expected, got)

print("\n2. EQUATION OF STATE, on random liquid fields")
rng = np.random.default_rng(20260904)
n = 100
rhoml_val = rng.uniform(0.05, 1.0, n)
rhobl_val = rng.uniform(0.05, 1.0, n)
delta_pred = 0.7314

# --- original, lines 199-226 ---
rhoml_arr = rhoml_val * rhol_M_ini
rhobl_arr = rhobl_val * rhol_B_ini
sum_l_arr = rhoml_arr + rhobl_arr
wt_M_arr = rhoml_arr / sum_l_arr
wt_B_arr = 1.0 - wt_M_arr
rhol_val_dynamic = 1.0 / (wt_M_arr / rho_M_pure + wt_B_arr / rho_B_pure)
scaled_rhol_current = rhol_val_dynamic / rhol_0
x_M_arr = (wt_M_arr / Mm) / (wt_M_arr / Mm + wt_B_arr / Mb)
x_B_arr = 1.0 - x_M_arr
Ml_arr = (x_M_arr * Mm) + (x_B_arr * Mb)
scaled_Ml_arr = Ml_arr / Ml0
Dl_arr = (Dmb0 ** x_B_arr) * (Dbm0 ** x_M_arr)
eff_Dl_arr = (Dl_arr / Dmb0) / delta_pred
K_part = scaled_rhol_current[-1] / scaled_Ml_arr[-1]

# --- package ---
wt_M_new, wt_B_new = mix.weight_fractions(rhoml_val, rhobl_val)
rhol_new, scaled_rhol_new = mix.scaled_density(wt_M_new, wt_B_new)
x_M_new, x_B_new = mix.mole_fractions(wt_M_new, wt_B_new)
scaled_Ml_new = mix.scaled_molar_mass(x_M_new, x_B_new)
eff_Dl_new = mix.effective_liquid_diffusivity(x_M_new, x_B_new, delta_pred)
K_part_new = mix.partition_coefficient(scaled_rhol_new[-1], scaled_Ml_new[-1])

check("weight_fractions (M)", wt_M_arr, wt_M_new)
check("weight_fractions (B)", wt_B_arr, wt_B_new)
check("scaled_density (unscaled)", rhol_val_dynamic, rhol_new)
check("scaled_density (scaled)", scaled_rhol_current, scaled_rhol_new)
check("mole_fractions (M)", x_M_arr, x_M_new)
check("scaled_molar_mass", scaled_Ml_arr, scaled_Ml_new)
check("effective_liquid_diffusivity (Vignes)", eff_Dl_arr, eff_Dl_new)
check("partition_coefficient", K_part, K_part_new)

Lg_pred = SCALED_B - delta_pred
check("gas_diffusivity (M)", (1.0 / alpha_M) / Lg_pred,
      mix.gas_diffusivity(mix.alpha_A, Lg_pred))
check("stefan_factor k", 1.0 - const_rho * scaled_rhol_current[-1],
      mix.stefan_factor(scaled_rhol_new[-1]))


class _Mesh:
    """Only the dx-dependent operators the balances actually use."""

    def __init__(self, dx):
        self.dx = dx

    def boundary_coefficient(self, D):
        return 2.0 * D / self.dx


print("\n3. LOCAL FLUX BALANCE, 2000 random draws")
local = LocalFluxBalance(mix, _Mesh(dx))
worst_v0 = worst_uN = 0.0
for _ in range(2000):
    g, D, rl, rg, Cl, K, dd, sr = rng.uniform(1e-5, 2.0, 8)
    # original, lines 104-112
    C_gas_bnd = 2.0 * D / dx
    numerator = Cl * rl + g * C_gas_bnd * rg
    denominator = (g * C_gas_bnd + Cl * K
                   + dd * (K - g * const_rho * sr))
    v0_ref = numerator / denominator
    uN_ref = K * v0_ref
    v0_new, uN_new = local.solve(g, D, rl, rg, Cl, K, dd, sr)
    worst_v0 = max(worst_v0, abs(v0_ref - v0_new))
    worst_uN = max(worst_uN, abs(uN_ref - uN_new))
check("interface v0", 0.0, worst_v0)
check("interface uN", 0.0, worst_uN)

print("\n4. GLOBAL FLUX BALANCE, 2000 random draws")
glob = GlobalFluxBalance(mix)
worst_t1 = worst_t2 = worst_rate = 0.0
for _ in range(2000):
    om, ga, al, Lg, gr, dI = rng.uniform(1e-5, 2.0, 6)
    # original, lines 125-129
    t1_ref = (om * ga / (al * Lg)) * gr
    t2_ref = om * ga * Lg * dI
    t1_new, t2_new = glob.rate_terms(om, ga, al, Lg, gr, dI)
    worst_t1 = max(worst_t1, abs(t1_ref - t1_new))
    worst_t2 = max(worst_t2, abs(t2_ref - t2_new))

    Lg2, sr2, gM, dIM, IgM, gB, dIB, IgB = rng.uniform(0.5, 2.0, 8)
    # original, lines 275-284
    a1, a2 = glob.rate_terms(omega0_M, gamma_M, alpha_M, Lg2, gM, dIM)
    b1, b2 = glob.rate_terms(omega0_B, gamma_B, alpha_B, Lg2, gB, dIB)
    denom_ref = sr2 - (omega0_M * gamma_M * IgM) - (omega0_B * gamma_B * IgB)
    rate_ref = (a1 - a2 + b1 - b2) / denom_ref
    rate_new = glob.ddelta_dt(Lg2, sr2, gM, dIM, IgM, gB, dIB, IgB)
    worst_rate = max(worst_rate, abs(rate_ref - rate_new))
check("rate_terms term1", 0.0, worst_t1)
check("rate_terms term2", 0.0, worst_t2)
check("ddelta_dt", 0.0, worst_rate)

print("\n5. t=0 PROFILE COLUMNS are exactly ones/zeros")
ones = np.ones(100)
wt0_M, wt0_B = mix.weight_fractions(ones, ones)
_, scaled0 = mix.scaled_density(wt0_M, wt0_B)
check("scaled_rhol at t=0 == 1.0", ones, scaled0)

print("\n6. COMPONENT LIBRARY")
check("library methanol M", Mm, LIB.component("methanol").M)
check("library methanol p_sat", pm0, LIB.component("methanol").p_sat)
check("library butanol M", Mb, LIB.component("1-butanol").M)
check("override p_sat -> 0.0", 0.0, LIB.component("1-butanol", p_sat=0.0).p_sat)
check("library butanol p_sat (real value)", 887.0, LIB.component("1-butanol").p_sat)

D_ref, D_bm = LIB.binary_diffusivities("methanol", "1-butanol")
check("binary_diffusivities D_ref", Dmb0, D_ref)
check("binary_diffusivities D_bm", Dbm0, D_bm)
rev_ref, rev_bm = LIB.binary_diffusivities("1-butanol", "methanol")
check("binary_diffusivities reversed", [Dbm0, Dmb0], [rev_ref, rev_bm])

check("Mixture picks up D_ref from the pair", Dmb0, mix.D_ref)
check("Mixture picks up D_bm from the pair", Dbm0, mix.D_bm)

for label, call in [
        ("unknown component raises KeyError",
         lambda: LIB.component("kryptonite")),
        ("unknown pair raises KeyError",
         lambda: LIB.binary_diffusivities("water", "ethanol"))]:
    try:
        call()
    except KeyError:
        print(f"   PASS  {label:<44} raised")
    else:
        print(f"   FAIL  {label:<44} did NOT raise")
        failures.append(label)

# A different pair must build once its diffusivities are supplied explicitly.
other = Mixture(Component("water", 18.015, 997.0, 3169.0, 2.5e-5),
                Component("ethanol", 46.07, 785.1, 7870.0, 1.2e-5),
                0.4, R=R, T=T, rho_gas=rhog_0,
                D_ref=1.2e-9, D_bm=0.9e-9)
ok = other.rhol_0 > 0 and other.gamma_A > 0 and other.alpha_A > 0
print(f"   {'PASS' if ok else 'FAIL'}  {'water/ethanol mixture builds':<44} "
      f"rhol_0 = {other.rhol_0:.1f} kg/m3")
if not ok:
    failures.append("water/ethanol mixture builds")

print("\n6b. CUSTOM COMPONENT LIBRARY FROM DICTIONARIES")
ACETONE = {"M": 58.08, "rho_pure": 784.5, "p_sat": 30800.0, "Dg": 1.06e-5}
custom = ComponentLibrary(
    {"acetone": ACETONE, "water": {"M": 18.015, "rho_pure": 997.0, "p_sat": 3169.0, "Dg": 2.5e-5}},
    {("acetone", "water"): 1.28e-9, ("water", "acetone"): 4.56e-9},
)
check("custom library component M", ACETONE["M"],
      custom.component("acetone").M)
check("custom library D_ref", 1.28e-9,
      custom.binary_diffusivities("acetone", "water")[0])
check("custom library D_bm", 4.56e-9,
      custom.binary_diffusivities("acetone", "water")[1])
print(f"   PASS  {'custom library names()':<44} {custom.names()}")

# A Mixture must take its diffusivities from the library it is given, not from
# the package defaults.
custom_mix = Mixture(custom.component("acetone"), custom.component("water"),
                     0.5, R=R, T=T, rho_gas=rhog_0, library=custom)
check("Mixture(library=custom) D_ref", 1.28e-9, custom_mix.D_ref)
check("Mixture(library=custom) D_bm", 4.56e-9, custom_mix.D_bm)

# ...and the default library must be unaffected by any of this.
check("caller library untouched", Dmb0,
      LIB.binary_diffusivities("methanol", "1-butanol")[0])

# Extending rather than replacing.
extended = ComponentLibrary({**COMPONENTS, "acetone": ACETONE},
                            {**DIFFUSIVITIES,
                             ("acetone", "water"): 1.28e-9,
                             ("water", "acetone"): 4.56e-9})
ok = ("acetone" in extended.names() and "methanol" in extended.names())
print(f"   {'PASS' if ok else 'FAIL'}  {'extended library keeps defaults':<44} "
      f"{len(extended.names())} components")
if not ok:
    failures.append("extended library keeps defaults")

# Library copies its inputs, so mutating the source dict must not leak in.
src = {"acetone": dict(ACETONE)}
lib_copy = ComponentLibrary(src, {})
src["acetone"]["M"] = 999.0
check("library copies its input dicts", ACETONE["M"],
      lib_copy.component("acetone").M)

# Validation: a missing required key must fail at construction, not later.
for label, bad in [("missing Dg raises KeyError",
                    {"x": {"M": 1.0, "rho_pure": 1.0, "p_sat": 1.0}}),
                   ("missing p_sat raises KeyError",
                    {"x": {"M": 1.0, "rho_pure": 1.0, "Dg": 1.0}})]:
    try:
        ComponentLibrary(bad)
    except KeyError:
        print(f"   PASS  {label:<44} raised")
    else:
        print(f"   FAIL  {label:<44} did NOT raise")
        failures.append(label)

# With no data shipped in the package, Mixture must demand diffusivities.
try:
    Mixture(METHANOL, BUTANOL, 0.5, R=R, T=T, rho_gas=rhog_0)
except ValueError:
    print(f"   PASS  {'Mixture without library/D_ref raises':<44} raised")
else:
    print(f"   FAIL  {'Mixture without library/D_ref raises':<44} did NOT raise")
    failures.append("Mixture without library/D_ref raises")

try:
    ComponentLibrary({})
except ValueError:
    print(f"   PASS  {'empty library raises ValueError':<44} raised")
else:
    print(f"   FAIL  {'empty library raises ValueError':<44} did NOT raise")
    failures.append("empty library raises ValueError")

# The constants are inputs now: omitting one must be a TypeError, not a
# silently-applied default.
for label, call in [
        ("Mixture without R raises TypeError",
         lambda: Mixture(METHANOL, BUTANOL, 0.5, T=T, rho_gas=rhog_0,
                         library=LIB)),
        ("Mixture without rho_gas raises TypeError",
         lambda: Mixture(METHANOL, BUTANOL, 0.5, R=R, T=T, library=LIB)),
        ("R/T/rho_gas are keyword-only",
         lambda: Mixture(METHANOL, BUTANOL, 0.5, R, T, rhog_0, library=LIB))]:
    try:
        call()
    except TypeError:
        print(f"   PASS  {label:<44} raised")
    else:
        print(f"   FAIL  {label:<44} did NOT raise")
        failures.append(label)

print("\n7. INITIAL-PROFILE INPUT")
try:
    from solvent_evaporation.solver import _per_component
except ImportError as exc:                      # FiPy absent
    print(f"   SKIPPED (needs FiPy: {exc})")
else:
    check("scalar applies to both", [1.0, 1.0], list(_per_component(1.0)))
    check("2-tuple splits A/B", [0.9, 1.1], list(_per_component((0.9, 1.1))))
    prof = np.linspace(0.5, 1.5, 8)
    got_A, got_B = _per_component(prof)
    check("array applies to both", prof, got_A)
    check("array applies to both (B)", prof, got_B)
    try:
        _per_component((1.0, 2.0, 3.0))
    except ValueError:
        print(f"   PASS  {'3-tuple raises ValueError':<44} raised")
    else:
        print(f"   FAIL  {'3-tuple raises ValueError':<44} did NOT raise")
        failures.append("3-tuple raises ValueError")

    # The t=0 scaled density must follow a per-component profile, not be
    # assumed uniform: 0.9/1.1 is butanol-richer, so denser than the reference.
    wt_a, wt_b = mix.weight_fractions(0.9 * np.ones(4), 1.1 * np.ones(4))
    _, scaled_asym = mix.scaled_density(wt_a, wt_b)
    ok = float(scaled_asym[0]) > 1.0
    print(f"   {'PASS' if ok else 'FAIL'}  "
          f"{'asymmetric liquid_ini -> scaled_rhol != 1':<44} "
          f"{float(scaled_asym[0]):.6f}")
    if not ok:
        failures.append("asymmetric liquid_ini")

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED -- the package reproduces the original exactly")
