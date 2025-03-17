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

result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
    formalism="Nagaitsev",  # can also be "B&M"
    total_beam_intensity=bunch_intensity,
    emittance_coupling_factor=emittance_coupling_factor,
    emittance_constraint="coupling",  # can also be None or "excitation"
)

# TODO: add a print of the table and comment with top rows so users know what's there

######################################
# Comparison with analytical results #
######################################

factor = 1 + emittance_coupling_factor * (tw.partition_numbers[1] / tw.partition_numbers[0])

print()
print("Emittance constraint: coupling")
print("Horizontal steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {result.gemitt_x[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0] * factor))}")
print(f"ODE: {result.eq_sr_ibs_gemitt_x}")
print("Vertical steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {result.gemitt_y[0] / (1 - result.Kx[-1] / (tw.damping_constants_s[0] * factor))}")
print(f"ODE: {result.eq_sr_ibs_gemitt_y}")
print("Longitudinal steady-state emittance:")
print("-------------------------------")
print(f"Analytical: {result.gemitt_zeta[0] / (1 - result.Kz[-1] / (tw.damping_constants_s[2]))}")
print(f"ODE: {result.eq_sr_ibs_gemitt_zeta}")

# TODO: add print here with the outputs

########
# Plot #
########

# TODO: make sure plot code is not in the example
# TODO: add an hline with the analytical result

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained")

ax0.plot(result.time * 1e3, result.gemitt_x * 1e12, label=r"$\tilde{\varepsilon}_x$")
ax0.plot(result.time * 1e3, result.gemitt_y * 1e12, label=r"$\tilde{\varepsilon}_y$", ls="--")
ax0b = ax0.twinx()
ax0b.plot(result.time * 1e3, result.gemitt_zeta * 1e6, color="C2", label=r"$\varepsilon_z$")
ax1.plot(result.time * 1e3, result.Kx, label=r"$\alpha_x^{IBS}$")
ax1.plot(result.time * 1e3, result.Ky, label=r"$\alpha_y^{IBS}$")
ax1.plot(result.time * 1e3, result.Kz, color="C2", label=r"$\alpha_z^{IBS}$")

lines, labels = ax0.get_legend_handles_labels()
lines2, labels2 = ax0b.get_legend_handles_labels()
ax0.legend(lines + lines2, labels + labels2, ncol=3)
ax1.legend(ncol=3)
ax1.set_xlabel("Time [ms]")
ax0.set_ylabel(r"$\tilde{\varepsilon}$ [pm.rad]")
ax0b.set_ylabel(r"$\varepsilon_z$ [m]")
ax1.set_ylabel(r"$\alpha^{IBS}$ [s$^{-1}$]")
