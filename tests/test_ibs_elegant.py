import pathlib
import pytest
import xobjects as xo
import xtrack as xt
from ibs_conftest import XTRACK_TEST_DATA

result_elegant = {
    # Emittance coupling factor
    "0.02" : {
        "eps_x": 1.88319e-10,
        "sigma_z": 0.00491982,
        "sigma_delta": 0.00140017,
        "T_x": 106.648,
        "T_z": 32.8772
  },
    # Emittance coupling factor
    "0.1" : {
        "eps_x": 1.46119e-10,
        "sigma_z": 0.00441742,
        "sigma_delta": 0.00125719,
        "T_x": 81.5391,
        "T_z": 25.2408
  },
    # Emittance coupling factor
    "0.5" : {
        "eps_x": 1.05835e-10,
        "sigma_z": 0.00404997,
        "sigma_delta": 0.00115261,
        "T_x": 63.7394,
        "T_z": 17.7664
  },
    # Emittance coupling factor
    "1" : {
        "eps_x": 8.5549e-11,
        "sigma_z": 0.00395442,
        "sigma_delta": 0.00112542,
        "T_x": 63.1134,
        "T_z": 15.4736
  }
}


BUNCH_INTENSITY: float = 6.2e9  # 1C bunch intensity

# ----- Fixture for the (configured) BESSY III line ----- #


@pytest.fixture(scope="module")
def bessy3_line_with_radiation() -> xt.Line:
    """
    Loads the BESSY III lattice as a Line and
    configures radiation before returning it.
    """
    # -------------------------------------------
    # Load the line with a particle_ref
    bess3_dir = XTRACK_TEST_DATA / "bessy3"
    linefile = bess3_dir / "bessy3.json"
    line = xt.Line.from_json(linefile)
    # -------------------------------------------
    # Build tracker and configure radiation
    line.build_tracker()
    line.matrix_stability_tol = 1e-2
    line.configure_radiation(model="mean")
    line.compensate_radiation_energy_loss()
    # Run twiss in fixture to compile kernels once
    line.twiss(eneloss_and_damping=True)
    return line


# ----- Test Functions vs Analytical Formulae ----- #


@pytest.mark.parametrize("emittance_coupling_factor", [0.02, 0.1, 0.5, 1])
def test_equilibrium_vs_elegant(
    emittance_coupling_factor, bessy3_line_with_radiation: xt.Line
):
    """
    Load the BESSY III line and compute ierations until we reach
    an equilibrium with SR and IBS, in the case where we enforce
    a betatron coupling constraint on the transverse planes. The
    resulting values are tested against elegant results under the
    same conditions.
    """
    # -------------------------------------------
    # Get the twiss with SR effects from the configured line
    tw = bessy3_line_with_radiation.twiss(eneloss_and_damping=True)
    # -------------------------------------------
    # Compute the equilibrium emittances - coupling constraint
    result = tw.compute_equilibrium_emittances_from_sr_and_ibs(
        formalism="Nagaitsev",  # No Dy in the line, faster
        total_beam_intensity=BUNCH_INTENSITY,
        emittance_coupling_factor=emittance_coupling_factor,
        emittance_constraint="coupling",
    )
    # -------------------------------------------
    # Check xsuite results vs elegant results
    # Check the horizontal equilibrium emittance and IBS growth rate
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x, result_elegant[str(emittance_coupling_factor)]["eps_x"],
        atol=1e-12, rtol=5e-2,
    )
    # Factor of 2 because different conventions between Xsuite and elegant!
    xo.assert_allclose(
        result.Kx[-1], result_elegant[str(emittance_coupling_factor)]["T_x"] / 2,
        rtol=6e-2,
    )
    # Check the longitudinal equilibrium emittance and IBS growth rate
    # Different eps_zeta convention between both codes
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta, 
        result_elegant[str(emittance_coupling_factor)]["sigma_delta"] * result_elegant[str(emittance_coupling_factor)]["sigma_z"],
        rtol=5e-2,
    )
    # Factor of 2 because different conventions between Xsuite and elegant!
    xo.assert_allclose(
        result.Kz[-1], result_elegant[str(emittance_coupling_factor)]["T_z"] / 2,
        rtol=6e-2,
    )

    # -------------------------------------------
    # These are hardcoded (from ELEGANT version with
    # corrected partition numbers use)