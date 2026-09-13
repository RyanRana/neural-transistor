"""Real deployment targets, with sourced numbers and honest availability.

Power and memory figures are from datasheets, tagged by provenance, because vendor
marketing and the datasheet frequently disagree and the difference decides whether a
robot flies. Two entries carry availability warnings that matter more than their specs:

- **GAP9**: GreenWaves Technologies entered judicial liquidation on 14 Jan 2025. The
  part is excellent and not purchasable. The NE16 accelerator RTL survives open-source
  at pulp-platform/ne16.
- **Loihi 2**: no public SKU, INRC signup inactive, and every lava-nc repository was
  archived 2026-05-13. Smallest form factor (Kapoho Point) is 108 g -- 3.6x an entire
  sub-30 g robot's mass budget before the host board.

The `int8_contract` field is the one a compiler cares about. CMSIS-NN is bit-exact with
the TFLite Micro reference kernels -- int8 weights in [-127,127] with zero-point 0,
per-axis on conv; int8 activations per-tensor with a zero-point; int32 per-axis bias --
and ST Edge AI, Ambiq's neuralSPOT, ExecuTorch's Cortex-M backend and GAP9's NE16 all
speak the same contract. One emitter covers all of them. ESP32-S3 does not: it is
per-tensor symmetric power-of-two only, which is strictly more lossy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Device:
    name: str
    sram_kb: int
    flash_kb: int
    clock_mhz: int
    active_mw: float | None        # datasheet active power, full clock
    int8_contract: str             # "cmsis-nn" | "per-tensor-pow2" | "uint8-asym" | "none"
    note: str
    available: bool = True
    source: str = ""

    @property
    def flash_b(self) -> int:
        return self.flash_kb * 1024

    @property
    def sram_b(self) -> int:
        return self.sram_kb * 1024

    def fits(self, flash_b: int, ram_b: int) -> bool:
        return flash_b <= self.flash_b and ram_b <= self.sram_b


DEVICES: dict[str, Device] = {d.name: d for d in [
    Device("STM32F405", 196, 1024, 168, 132.0, "cmsis-nn",
           "The Crazyflie 2.x flight controller. The only MCU in this table proven "
           "airborne on a sub-30 g robot.", True, "DS8626"),
    Device("STM32F401", 96, 512, 84, 41.0, "cmsis-nn",
           "Smallest credible target. 96 KB SRAM is the real constraint.", True, "DS10086"),
    Device("STM32H743", 1060, 2048, 480, 363.0, "cmsis-nn",
           "Cortex-M7, double FPU. Fast but power-hungry for this class.", True, "DS12110"),
    Device("STM32U575", 786, 2048, 160, 22.1, "cmsis-nn",
           "Ultra-low-power line with SMPS. Best mW-per-neuron here. "
           "Cortex-M33 + DSP but no Helium.", True, "DS13737"),
    Device("nRF52840", 256, 1024, 64, 15.0, "cmsis-nn",
           "Cortex-M4F with BLE. Common on tiny robots for the radio, not the compute.",
           True, "nRF52840 PS v1.8"),
    Device("RP2350", 520, 4096, 150, 38.7, "cmsis-nn",
           "Dual M33. Has DSP but NOT the MVE vector extension, and SMLAD needs adjacent "
           "halfwords -- gathers defeat it, so budget ~1 MAC/cycle on sparse CSR.",
           True, "RP2350 datasheet"),
    Device("ESP32-S3", 416, 8192, 240, 330.0, "per-tensor-pow2",
           "Large external flash, but quantization is per-tensor symmetric power-of-two "
           "only (per-channel exists on P4/S3-N32R8, not plain S3) -- strictly lossier.",
           True, "ESP32-S3 datasheet"),
    Device("Apollo510", 3750, 4096, 250, 11.7, "cmsis-nn",
           "Most SRAM and lowest power in the table. Helium gives 8 int8 MAC/cycle. "
           "No NPU -- Ambiq states this explicitly.", True, "Apollo510 DS"),
    Device("Apollo4Plus", 2750, 2048, 192, 7.6, "cmsis-nn",
           "Sub-10 mW class with 2.7 MB SRAM.", True, "Apollo4 Plus DS"),
    Device("GAP9", 1728, 2048, 370, 45.0, "cmsis-nn",
           "RISC-V PULP cluster + NE16. Architecturally ideal and COMMERCIALLY DEAD: "
           "GreenWaves entered judicial liquidation 14 Jan 2025.", False, "DS + societe.com"),
]}

#: Chips proven to have run onboard inference on a real sub-30 g flying robot.
FLOWN_SUB30G = {
    "STM32F405": "Crazyflie 2.x, 27-29 g (product)",
    "GAP8": "PULP-Dronet on Crazyflie, 27 g, 64 mW, 3.5% of the power budget "
            "(arXiv:1805.01831) -- the only sub-30 g onboard DNN inference on record",
}

#: Power budgets measured on real platforms. (mass_g, total_mW, compute_mW)
POWER_BUDGETS = {
    "RoboBee":        (0.08, 19.0, 10.0),     # compute ~53% and still does not fit
    "Crazyflie 2.1+": (29.0, 1830.0, 64.0),   # compute 3.5%
}


def fit_report(flash_b: int, ram_b: int, include_unavailable: bool = True) -> dict:
    out = {}
    for name, d in DEVICES.items():
        if not include_unavailable and not d.available:
            continue
        out[name] = {
            "fits": d.fits(flash_b, ram_b),
            "flash_ok": flash_b <= d.flash_b, "ram_ok": ram_b <= d.sram_b,
            "flash_kb": d.flash_kb, "sram_kb": d.sram_kb,
            "active_mw": d.active_mw, "available": d.available,
            "int8_contract": d.int8_contract, "note": d.note,
        }
    return out
