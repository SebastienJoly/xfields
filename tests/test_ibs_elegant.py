import pathlib
import pytest
import pandas as pd
import xobjects as xo
import xtrack as xt
from ibs_conftest import XTRACK_TEST_DATA

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

fname = test_data_folder / "ibs_evolution.txt"
df = pd.read_csv(fname, sep='\t')

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
    # Load elegant results for this scenario 
    result_elegant = df.query(f"Coupling == {emittance_coupling_factor}")
    # -------------------------------------------
    # Check xsuite results vs elegant results
    # Check the horizontal equilibrium emittance and IBS growth rate
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_x, result_elegant.eps_x.iloc[-1],
        atol=1e-12, rtol=5e-2,
    )
    # Factor of 2 because different conventions between Xsuite and elegant!
    xo.assert_allclose(
        result.Kx[-1], result_elegant.T_x.iloc[-1] / 2,
        rtol=6e-2,
    )
    # Check the longitudinal equilibrium emittance and IBS growth rate
    # Different eps_zeta convention between both codes
    xo.assert_allclose(
        result.eq_sr_ibs_gemitt_zeta, 
        result_elegant.sigma_delta.iloc[-1] * result_elegant.sigma_z.iloc[-1],
        rtol=5e-2,
    )
    # Factor of 2 because different conventions between Xsuite and elegant!
    xo.assert_allclose(
        result.Kz[-1], result_elegant.T_z.iloc[-1] / 2,
        rtol=6e-2,
    )

    # -------------------------------------------
    # These are hardcoded (from ELEGANT version with
    # corrected partition numbers use)