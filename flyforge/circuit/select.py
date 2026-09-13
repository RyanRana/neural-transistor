"""Picking a set of neurons out of 211,577 without writing pandas by hand.

A Selector is a declarative predicate over the annotation table. They compose with
``|`` (union), ``&`` (intersection) and ``-`` (difference), so a circuit definition
reads as the anatomy it means:

    compass = Sel.type(r"^(EPG|PEN|PEG|Delta7|ER)") & Sel.side("L")
    front_leg = Sel.neuromere("T1") & (Sel.superclass("vnc_intrinsic") | Sel.motor())

Type patterns are regexes anchored with ``match`` because Janelia type names are
prefix-structured ("PFNp_b", "T4c", "MBON01", "DNa02") and prefix is the biologically
meaningful unit -- ``^T4`` is "all T4 subtypes" and that is almost always what is meant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Selector:
    fn: Callable[[pd.DataFrame], np.ndarray]
    label: str = "sel"

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.fn(df), dtype=bool)

    def __or__(self, o): return Selector(lambda d: self(d) | o(d), f"({self.label}|{o.label})")
    def __and__(self, o): return Selector(lambda d: self(d) & o(d), f"({self.label}&{o.label})")
    def __sub__(self, o): return Selector(lambda d: self(d) & ~o(d), f"({self.label}-{o.label})")
    def __invert__(self): return Selector(lambda d: ~self(d), f"~{self.label}")

    def idx(self, df: pd.DataFrame) -> np.ndarray:
        """Dense neuron indices matching, sorted."""
        return np.sort(df.loc[self(df), "idx"].to_numpy().astype(np.int32))


class Sel:
    """Constructors for the common predicates."""

    @staticmethod
    def type(pattern: str) -> Selector:
        return Selector(lambda d: d["type"].str.match(pattern, na=False), f"type~{pattern}")

    @staticmethod
    def type_in(names: Iterable[str]) -> Selector:
        s = set(names)
        return Selector(lambda d: d["type"].isin(s), f"type in {len(s)}")

    @staticmethod
    def superclass(*names: str) -> Selector:
        s = set(names)
        return Selector(lambda d: d["superclass"].isin(s), f"super={'/'.join(names)}")

    @staticmethod
    def neuromere(*names: str) -> Selector:
        s = set(names)
        return Selector(lambda d: d["somaNeuromere"].isin(s), f"nm={'/'.join(names)}")

    @staticmethod
    def side(*names: str) -> Selector:
        s = set(names)
        return Selector(lambda d: d["somaSide"].isin(s), f"side={'/'.join(names)}")

    @staticmethod
    def nt(*names: str) -> Selector:
        s = {n.lower() for n in names}
        return Selector(lambda d: d["consensus_nt"].astype("string").str.lower().isin(s),
                        f"nt={'/'.join(names)}")

    @staticmethod
    def motor() -> Selector:
        return Sel.superclass("vnc_motor", "cb_motor")

    @staticmethod
    def sensory() -> Selector:
        return Sel.superclass("vnc_sensory", "cb_sensory", "ol_sensory", "sensory_ascending")

    @staticmethod
    def descending() -> Selector:
        return Sel.superclass("descending_neuron")

    @staticmethod
    def ascending() -> Selector:
        return Sel.superclass("ascending_neuron")

    @staticmethod
    def traced(min_quality: str = "Roughly traced") -> Selector:
        """Drop orphans, glia and out-of-scope fragments."""
        good = {"Reviewed", "Roughly traced", "Prelim Roughly traced", "RT Hard to trace"}
        if min_quality == "Reviewed":
            good = {"Reviewed"}
        return Selector(lambda d: d["statusLabel"].isin(good), f"traced>={min_quality}")

    @staticmethod
    def all() -> Selector:
        return Selector(lambda d: np.ones(len(d), dtype=bool), "all")

    @staticmethod
    def none() -> Selector:
        return Selector(lambda d: np.zeros(len(d), dtype=bool), "none")
