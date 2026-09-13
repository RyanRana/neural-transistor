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
    """The connectome, or a clean skip.

    Most of this file asserts numbers that only exist once the 1.1 GB release is on
    disk. Erroring when it is absent makes a machine without the data look like a
    machine with a broken build, which is what it did in CI. Skipping says the true
    thing: these assertions were not checked here.
    """
    try:
        return Connectome.load(verbose=False)
    except FileNotFoundError as e:
        pytest.skip(f"connectome release not present: {e}")


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


#: The two shapes every real URDF has and the toy example does not: the robot hangs
#: off a base_footprint through a fixed joint, each hip carries a fixed decoration,
#: and the limb forks into two actuated branches at the end.
_REAL_SHAPED_URDF = """<?xml version="1.0"?>
<robot name="shaped">
  <link name="base_footprint"/><link name="chassis"/>
  <joint name="chassis_joint" type="fixed">
    <parent link="base_footprint"/><child link="chassis"/></joint>
  <link name="c1_rf"/><link name="c1_rf_shell"/><link name="thigh_rf"/><link name="tibia_rf"/>
  <joint name="j_c1_rf" type="revolute"><parent link="chassis"/><child link="c1_rf"/>
    <axis xyz="0 0 1"/></joint>
  <joint name="shell_rf" type="fixed"><parent link="c1_rf"/><child link="c1_rf_shell"/></joint>
  <joint name="j_thigh_rf" type="revolute"><parent link="c1_rf"/><child link="thigh_rf"/>
    <axis xyz="0 1 0"/></joint>
  <joint name="j_tibia_rf" type="revolute"><parent link="thigh_rf"/><child link="tibia_rf"/>
    <axis xyz="0 1 0"/></joint>
  <link name="wrist"/><link name="leftfinger"/><link name="rightfinger"/>
  <joint name="wrist_joint" type="fixed"><parent link="tibia_rf"/><child link="wrist"/></joint>
  <joint name="finger_joint1" type="prismatic"><parent link="wrist"/>
    <child link="leftfinger"/><axis xyz="0 1 0"/></joint>
  <joint name="finger_joint2" type="prismatic"><parent link="wrist"/>
    <child link="rightfinger"/><axis xyz="0 1 0"/></joint>
</robot>
"""


def test_urdf_import_looks_through_fixed_joints(tmp_path):
    """A fixed joint is a rigid offset, never a limb boundary.

    Stopping at the first one imported five of eight real robots (PhantomX, Minitaur,
    Husky, racecar, Crazyflie) as zero limbs, and the Unitree A1 as four one-DOF legs
    instead of four three-DOF legs -- the "any robot" promise failing silently.
    """
    p = tmp_path / "shaped.urdf"
    p.write_text(_REAL_SHAPED_URDF)
    spec, rep = urdf.parse(p)

    leg = next(l for l in spec.limbs if l.name == "c1_rf")
    assert leg.joints == ["j_c1_rf", "j_thigh_rf", "j_tibia_rf"]
    assert leg.joint_roles[:3] == ["coxa_yaw", "trochanter", "knee"]
    # the fork into two fingers ends the leg and starts two more limbs, rather than
    # being swallowed or dropped
    assert {l.name for l in spec.limbs} == {"c1_rf", "leftfinger", "rightfinger"}
    # the invariant that makes a silent miss impossible to ship
    assert rep["n_actuated_in_limbs"] == rep["n_actuated"] == 5


def test_side_and_segment_are_tokens_not_substrings():
    """``_l`` is in the word "link", and ``1`` is in the PhantomX link name ``c1_rr``."""
    assert urdf._infer_side("front_left_wheel_link") == "L"
    assert urdf._infer_side("rear_right_wheel_link") == "R"
    assert urdf._infer_side("motor_front_rightR_link") == "R"   # camelCase hump
    assert urdf._infer_side("panda_leftfinger") == "L"
    assert urdf._infer_side("panda_link1") == ""                # an arm has no side

    # a hexapod whose legs are named right/left x front/middle/rear must land on the
    # fly's own six legs, one for one
    legs = [M.Limb(name=f"c1_{s}{p}", joints=[f"j_c1_{s}{p}"], joint_roles=["coxa_yaw"],
                   side=urdf._infer_side(f"c1_{s}{p}"), index=i)
            for i, (s, p) in enumerate([("r", "f"), ("r", "m"), ("r", "r"),
                                        ("l", "f"), ("l", "m"), ("l", "r")])]
    src = R.infer_sources(M.MorphologySpec("phantomx", legs))
    assert src == {"c1_rf": ("front", "R"), "c1_rm": ("middle", "R"),
                   "c1_rr": ("hind", "R"), "c1_lf": ("front", "L"),
                   "c1_lm": ("middle", "L"), "c1_lr": ("hind", "L")}


def test_gait_phases_follow_the_legs_not_the_list_order():
    """A tripod is a fact about which legs, not about which list positions.

    The builtin hexapod lists ``L1 R1 L2 R2 L3 R3``; PhantomX lists all three right
    legs and then all three left ones. Assigning the published phase vector by position
    puts both front legs in the same group, which is not a tripod.
    """
    legs = [M.Limb(name=f"c1_{s}{p}", joints=[f"j_c1_{s}{p}"], joint_roles=["coxa_yaw"],
                   side=s.upper(), index=i)
            for i, (s, p) in enumerate([("r", "f"), ("r", "m"), ("r", "r"),
                                        ("l", "f"), ("l", "m"), ("l", "r")])]
    morph = M.MorphologySpec("phantomx", legs)
    phases, name = R.infer_phases(morph, "tripod", R.infer_sources(morph))
    assert name == "tripod"
    # right-front, left-middle and right-hind step together; the other three alternate
    assert phases["c1_rf"] == phases["c1_lm"] == phases["c1_rr"]
    assert phases["c1_lf"] == phases["c1_rm"] == phases["c1_lr"]
    assert phases["c1_rf"] != phases["c1_lf"]

    quad = M.MorphologySpec("a1", [
        M.Limb(name=n, joints=[f"{n}_j"], joint_roles=["coxa_yaw"], side=n[1], index=i)
        for i, n in enumerate(["FR_hip", "FL_hip", "RR_hip", "RL_hip"])])
    ph, _ = R.infer_phases(quad, "trot", R.infer_sources(quad))
    assert ph["FL_hip"] == ph["RR_hip"] and ph["FR_hip"] == ph["RL_hip"]   # diagonals
    assert ph["FL_hip"] != ph["FR_hip"]


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
