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
emittance_coupling_factor = 1  # round beam

# If not providing starting emittances, the function will
# default to the SR equilibrium emittances from the TwissTable

result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
    formalism="Nagaitsev",  # can also be "B&M"
    total_beam_intensity=bunch_intensity,
    # gemitt_x=...,  # defaults to tw.eq_gemitt_x
    # gemitt_y=...,  # defaults to tw.eq_gemitt_x
    # gemitt_zeta=...,  # defaults to tw.eq_gemitt_zeta
    emittance_coupling_factor=emittance_coupling_factor,
    emittance_constraint="coupling",
)

# The returned object is a Table
print(result)
# TODO: output of the print

######################################
# Comparison with analytical results #
######################################

# These are analytical estimate (from the last step's IBS growth rates)
# The factor below is to be respected with the coupling constraint
factor = 1 + emittance_coupling_factor * (tw.partition_numbers[1] / tw.partition_numbers[0])
analytical_x = result.gemitt_x[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0] * factor))
analytical_y = result.gemitt_y[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0] * factor))
analytical_z = result.gemitt_zeta[0] / (1 - result.Kz[-1] / (tw.damping_constants_s[2]))

print("Emittance Constraint: Coupling")
print("Horizontal steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {analytical_x}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_x}")
print("Vertical steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {analytical_y}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_y}")
print("Longitudinal steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {analytical_z}")
print(f"ODE:        {result.eq_sr_ibs_gemitt_zeta}")

# TODO: add print here with the outputs

# The results from the table can easily be plotted to view
# at the evolution of various parameters across time steps

#!end-doc-part

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained")

(l1,) = ax0.plot(result.time * 1e3, result.gemitt_x * 1e12, ls="-", label=r"$\tilde{\varepsilon}_x$")
(l2,) = ax0.plot(result.time * 1e3, result.gemitt_y * 1e12, ls="--", label=r"$\tilde{\varepsilon}_y$")
(l4,) = ax0.axhline(analytical_x * 1e12, color="C0", ls="-.", label=r"Analytical $\gemitt_{x}^{eq}$")
(l5,) = ax0.axhline(analytical_y * 1e12, color="C1", ls="-.", label=r"Analytical $\gemitt_{y}^{eq}$")
ax0b = ax0.twinx()
(l3,) = ax0b.plot(result.time * 1e3, result.gemitt_zeta * 1e6, color="C2", label=r"$\varepsilon_z$")
(l4,) = ax0b.axhline(analytical_z * 1e12, color="C2", ls="-.", label=r"Analytical $\gemitt_{\zeta}^{eq}$")
ax0.legend(handles=[l1, l2, l3, l4, l5], ncols=2)

ax1.plot(result.time * 1e3, result.Kx, label=r"$\alpha_{x}^{IBS}$")
ax1.plot(result.time * 1e3, result.Ky, label=r"$\alpha_{y}^{IBS}$")
ax1.plot(result.time * 1e3, result.Kz, color="C2", label=r"$\alpha_{z^{IBS}$")
ax1.legend()

ax1.set_xlabel("Time [ms]")
ax0.set_ylabel(r"$\tilde{\varepsilon}$ [pm.rad]")
ax0b.set_ylabel(r"$\varepsilon_z$ [m]")
ax1.set_ylabel(r"$\alpha^{IBS}$ [s$^{-1}$]")
