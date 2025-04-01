# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #
import matplotlib.pyplot as plt
import xtrack as xt
from scipy.constants import e

##########################
# Load xt.Line from file #
##########################

fname_line_particles = "../../../xtrack/test_data/bessy3/bessy3.json"
line = xt.Line.from_json(fname_line_particles)  # has particle_ref
line.build_tracker()

########################
# Twiss with Radiation #
########################

# We need to Twiss with Synchrotron Radiation enabled to obtain
# the SR equilibrium emittances and damping constants

line.matrix_stability_tol = 1e-2
line.configure_radiation(model="mean")
line.compensate_radiation_energy_loss()
tw = line.twiss(eneloss_and_damping=True)

######################################
# Steady-State Emittance Calculation #
######################################

bunch_intensity = 1e-9 / e  # 1C bunch charge
emittance_coupling_factor = 0.5  # for excitation this time

# One can provide specific values for starting emittances,
# as well as sigma_zeta / sigma_delta. A specific time step
# or relative tolerance for convergence can also be provided.

gemitt_x = 1.1 * tw.eq_gemitt_x  # larger horizontal emittance
gemitt_y = 0.5 * tw.eq_gemitt_y  # smaller vertical emittance

# larger sigma_zeta and sigma_delta from potential well distortion for example
overwrite_sigma_zeta = 1.2 * (tw.eq_gemitt_zeta * tw.bets0) ** 0.5  # larger sigma_zeta
overwrite_sigma_delta = 1.2 * (tw.eq_gemitt_zeta / tw.bets0) ** 0.5  # larger sigma_delta

result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
    formalism="nagaitsev",  # can also be "bjorken-mtingwa"
    total_beam_intensity=bunch_intensity,
    gemitt_x=gemitt_x,  # provided explicitely
    gemitt_y=gemitt_y,  # provided explicitely
    overwrite_sigma_zeta=overwrite_sigma_zeta,  # will recompute gemitt_zeta
    overwrite_sigma_delta=overwrite_sigma_delta,  # will recompute gemitt_zeta
    emittance_coupling_factor=emittance_coupling_factor,
    emittance_constraint="excitation",
)

# The returned object is a Table
print(result)
# TODO: output of the print

######################################
# Comparison with analytical results #
######################################

# These are analytical estimate (from the last step's IBS growth rates)
analytical_x = result.gemitt_x[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0]))
analytical_y = result.gemitt_y[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0]))
analytical_z = result.gemitt_zeta[0] / (1 - result.Kz[-1] / (tw.damping_constants_s[2]))

print()
print("Emittance Constraint: Excitation")
print("Horizontal steady-state emittance:")
print("---------------------------------")
print(f"Analytical: {analytical_x}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_x}")
print("Vertical steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {analytical_y}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_y}")
print("Longitudinal steady-state emittance:")
print("-----------------------------------")
print(f"Analytical: {analytical_z}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_zeta}")

# TODO: add print here with the outputs

# The results from the table can easily be plotted to view
# at the evolution of various parameters across time steps

#!end-doc-part
# fmt: off

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained")

(l1,) = ax0.plot(result.time * 1e3, result.gemitt_x * 1e12, ls="-", label=r"$\tilde{\varepsilon}_x$")
(l2,) = ax0.plot(result.time * 1e3, result.gemitt_y * 1e12, ls="--", label=r"$\tilde{\varepsilon}_y$")
l4 = ax0.axhline(analytical_x * 1e12, color="C0", ls="-.", label=r"Analytical $\varepsilon_{x}^{eq}$")
l5 = ax0.axhline(analytical_y * 1e12, color="C1", ls="-.", label=r"Analytical $\varepsilon_{y}^{eq}$")
ax0b = ax0.twinx()
(l3,) = ax0b.plot(result.time * 1e3, result.gemitt_zeta * 1e6, color="C2", label=r"$\varepsilon_z$")
l6 = ax0b.axhline(analytical_z * 1e6, color="C2", ls="-.", label=r"Analytical $\varepsilon_{\zeta}^{eq}$")
ax0.legend(handles=[l1, l2, l3, l4, l5], ncols=2)

ax1.plot(result.time * 1e3, result.Kx, label=r"$\alpha_{x}^{IBS}$")
ax1.plot(result.time * 1e3, result.Ky, label=r"$\alpha_{y}^{IBS}$")
ax1.plot(result.time * 1e3, result.Kz, label=r"$\alpha_{z}^{IBS}$")
ax1.legend()

ax1.set_xlabel("Time [ms]")
ax0.set_ylabel(r"$\tilde{\varepsilon_{x,y}}$ [pm.rad]")
ax0b.set_ylabel(r"$\varepsilon_{\zeta}$ [m]")
ax1.set_ylabel(r"$\alpha^{IBS}$ [$s^{-1}$]")
fig.align_ylabels((ax0, ax1))
plt.tight_layout()
plt.show()
