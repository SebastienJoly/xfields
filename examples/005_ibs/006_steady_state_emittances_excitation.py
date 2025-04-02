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
# but we need to ensure they respect the emittance coupling
# contraint we want to enforce
gemitt_x = 1.1 * tw.eq_gemitt_x  # larger horizontal emittance
gemitt_y = emittance_coupling_factor * gemitt_x  # enforce the constraint

# One can overwrite sigma_zeta / sigma_delta (larger
# values from potential well distortion for example)
overwrite_sigma_zeta = 1.2 * (tw.eq_gemitt_zeta * tw.bets0) ** 0.5  # larger sigma_zeta
overwrite_sigma_delta = 1.2 * (tw.eq_gemitt_zeta / tw.bets0) ** 0.5  # larger sigma_delta

# A specific time step or relative tolerance for convergence can also be provided.
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


# The results from the table can easily be plotted to view
# at the evolution of various parameters across time steps

#!end-doc-part
# fmt: off

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, layout="constrained")

(l1,) = ax0.plot(result.time * 1e3, result.gemitt_x * 1e12, ls="-", label=r"$\tilde{\varepsilon}_x$")
(l2,) = ax0.plot(result.time * 1e3, result.gemitt_y * 1e12, ls="--", label=r"$\tilde{\varepsilon}_y$")
ax0b = ax0.twinx()
(l3,) = ax0b.plot(result.time * 1e3, result.gemitt_zeta * 1e6, color="C2", label=r"$\varepsilon_z$")
ax0.legend(handles=[l1, l2, l3], ncols=2)

ax1.plot(result.time * 1e3, result.Kx, label=r"$\alpha_{x}^{IBS}$")
ax1.plot(result.time * 1e3, result.Ky, label=r"$\alpha_{y}^{IBS}$")
ax1.plot(result.time * 1e3, result.Kz, label=r"$\alpha_{z}^{IBS}$")
ax1.legend()

ax1.set_xlabel("Time [ms]")
ax0.set_ylabel(r"$\tilde{\varepsilon}_{x,y}$ [pm.rad]")
ax0b.set_ylabel(r"$\varepsilon_{\zeta}$ [m]")
ax1.set_ylabel(r"$\alpha^{IBS}$ [$s^{-1}$]")
fig.align_ylabels((ax0, ax1))
plt.tight_layout()
plt.show()
