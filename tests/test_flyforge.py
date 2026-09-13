"""Tests that assert the claims in the README, so the README cannot rot silently."""
import numpy as np
import pytest

from neuraltransistor.circuit import noise
from neuraltransistor.circuit.extract import LIBRARY, from_library
from neuraltransistor.circuit.sign import assign_signs, naive_signs
from neuraltransistor.data.source import Connectome
from neuraltransistor.evals import suite
from neuraltransistor.ir.graph import CircuitIR
from neuraltransistor.morph import retarget as R, spec as M, urdf
from neuraltransistor.quant.quantize import compress, input_drive, prune


@pytest.fixture(scope="session")
def conn():
    return Connectome.load(verbose=False)


@pytest.fixture(scope="session")
def compass(conn):
    return from_library(conn, "compass")


def test_index_shape(conn):
    assert len(conn) == 211_577
    assert conn.meta.n_edges_total == 151_856_684
    assert conn.meta.n_synapses_total == 311_833_243
    assert conn.meta.max_weight == 2_591
    # the honest filter: few edges kept, but most synaptic mass
    assert 0.15 < conn.meta.retained < 0.20
    assert 0.38 < conn.meta.retained_synapses < 0.43


def test_weights_are_already_low_precision(conn):
    """Two populations, two numbers -- do not quote one for the other.

    Raw release table (151.9M edges): 62.0% single-synapse, 99.03% <= 15.
    Retained annotated graph (26.0M edges): 40.4% single-synapse, 94.3% <= 15,
    98.1% <= 31. Dropping unannotated fragments removes weak edges preferentially,
    so the graph you actually compile needs about one more bit than the raw
    distribution implies.
    """
    w = conn.data
    assert 0.38 < (w <= 1).mean() < 0.43    # retained graph, not the raw table
    assert 0.94 < (w <= 15).mean() < 0.95   # 4 bits covers 94.3%
    assert (w <= 31).mean() > 0.98          # 5 bits covers 98.1%
    assert w.max() == 2_591


def test_nmj_correction_changes_motor_signs(conn):
    mn = conn.neurons[conn.neurons.superclass == "vnc_motor"]
    sign, kind, rep = assign_signs(mn)
    naive = naive_signs(mn)
    assert rep.n_nmj_corrected > 100
    # the naive map calls hundreds of motor neurons inhibitory; the guard does not
    assert (naive == -1).sum() > 200
    assert (sign == -1).sum() < 50


def test_noise_model_is_a_real_mixture(conn):
    m = noise.get(conn)
    assert 0.2 < m.chance < 0.4          # shuffled control
    assert m.ceiling > 0.98              # strong edges reproduce
    p1 = float(m.p_real(np.array([1]))[0])
    p20 = float(m.p_real(np.array([20]))[0])
    assert p1 < 0.85 < p20               # single synapses are the suspect ones
    assert m.min_weight_for(0.95) >= m.min_weight_for(0.90)


def test_every_library_circuit_extracts(conn):
    for key in LIBRARY:
        ir = from_library(conn, key)
        assert ir.n_neurons > 0
        assert ir.indptr[-1] == ir.n_edges
        assert ir.sign.shape == (ir.n_neurons,)
        assert set(np.unique(ir.sign)).issubset({-1, 0, 1})


def test_modulatory_edges_are_split_out(conn):
    ir = from_library(conn, "gate")
    assert ir.n_mod_edges > 0
    # a modulatory neuron must not also appear in the chemical edge list
    mod = np.flatnonzero(ir.sign_kind == 3)
    src = np.repeat(np.arange(ir.n_neurons), np.diff(ir.indptr))
    assert not np.isin(src, mod).any()


def test_pruning_keeps_more_mass_than_edges(compass):
    out = prune(compass, 5)
    assert out.n_edges < compass.n_edges
    assert (out.n_synapses / compass.n_synapses) > (out.n_edges / compass.n_edges)


def test_int8_is_nearly_lossless_on_drive(compass):
    _, rep = compress(compass, weight_bits=8, prune_min_weight=1)
    assert rep.drive_rel_err_mean < 0.01
    assert rep.drive_sign_flips == 0


def test_roundtrip_preserves_fingerprint(compass, tmp_path):
    p = compass.save(tmp_path / "c.fcx")
    assert CircuitIR.load(p).fingerprint() == compass.fingerprint()


def test_emitted_c_matches_reference_bit_for_bit(compass):
    r = suite.eval_c_equivalence(compass, ticks=32)
    assert r.passed, r.detail


def test_emitted_c_is_warning_clean(compass):
    r = suite.eval_compiles(compass)
    assert r.passed, r.detail.get("stderr")


def test_motor_decode_covers_all_six_legs(conn):
    ir = from_library(conn, "legs")
    d = M.decode_motor(ir, conn.neurons)
    assert {x.leg for x in d} >= {"front", "middle", "hind"}
    assert {x.joint for x in d} >= {"coxa_yaw", "trochanter", "knee"}
    for name in ("hexapod", "quadruped", "biped"):
        assert M.coverage(d, M.BUILTIN[name]())["coverage"] == 1.0


def test_retarget_reports_what_it_drops(conn):
    ir = from_library(conn, "legs")
    d = M.decode_motor(ir, conn.neurons)
    rt = R.retarget(M.biped(), d)
    assert len(rt.bindings) == 2
    assert rt.unused_sources          # a biped cannot use four of the six legs
    assert rt.note


def test_urdf_import(tmp_path):
    p = urdf.write_example(tmp_path / "q.urdf")
    spec, rep = urdf.parse(p)
    assert rep["n_actuated"] == 12
    assert spec.n_limbs == 4
    assert all("knee" in l.joint_roles for l in spec.limbs)


def test_budget_solver_refuses_rather_than_lying(compass):
    from neuraltransistor.quant.quantize import fit_budget
    out, rep, trials = fit_budget(compass, budget_kb=0.5, max_drive_err=0.01)
    assert out is None and rep is None    # nothing fits 0.5 KB; must not fake it
    assert trials


def test_looming_reference_matches_published_bracket():
    """Our giant-fiber implementation must peak where the published model says.

    Ache 2019 Fig 4B brackets the peak: a pure size detector peaks when the delayed
    angular size crosses 42 deg (t = -2.6051*tau + 19 ms), a pure velocity detector when
    expansion peaks at 90 deg (t = -1.0*tau + 19 ms). A real GF sums both and lands
    between them. Outside that bracket means the implementation is wrong.
    """
    from neuraltransistor.stimuli import Looming, giant_fiber
    for lv in (0.010, 0.020, 0.040, 0.070, 0.100, 0.140):
        loom = Looming(l_over_v=lv, dt=0.0005)
        g = giant_fiber(loom)
        lo, hi = loom.peak_bracket_ms()
        assert lo <= g["peak_t_ms"] <= hi, (
            f"r/v={lv*1000:.0f}ms: peak {g['peak_t_ms']:.1f} outside [{lo:.1f},{hi:.1f}]")


def test_looming_stimulus_conventions():
    """Full angle, starts at 10 deg, expansion stops at exactly t = -tau."""
    import numpy as np
    from neuraltransistor.stimuli import Looming
    loom = Looming(l_over_v=0.040, dt=0.0005)
    assert abs(np.degrees(loom.angular_size[0]) - 10.0) < 0.5
    # t(90 deg) = -tau exactly
    assert abs(loom.t_at_size(90.0) - (-0.040)) < 1e-9
    assert abs(loom.t_at_size(39.0) / 0.040 + 2.8241) < 0.01


def test_circuits_have_engineering_roles():
    """Every circuit says what it DOES, not only what it is called in a fly."""
    from neuraltransistor.circuit.extract import LIBRARY
    for key, spec in LIBRARY.items():
        assert spec.get("role"), f"{key} has no engineering role"
        assert spec.get("does"), f"{key} has no plain-language description"
