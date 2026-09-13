"""A complete bootable firmware project around an emitted circuit.

``emit_c`` produces a kernel: three translation units with no ``main``, no reset vector
and no memory map. That is the right shape for dropping into someone's existing
firmware, and it is also why every size figure in this repo used to be datasheet
arithmetic -- ``sizeof(the arrays we wrote)`` compared against ``sizeof(the part)``.
Nothing had been through a linker, so nothing had a real ``.text``.

This module closes that. It writes startup code, a linker script, a ``main`` and a
Makefile around the emitted kernel, cross-builds it with ``arm-none-eabi-gcc``, and
reports the section sizes the linker actually produced. Paired with ``cilicon.yml`` it
also boots the result in an emulated Cortex-M and checks what it computed.

**The boot proof checks correctness, not just liveness.** ``main`` runs the circuit for a
fixed number of ticks against a fixed input pattern and folds every spike into an FNV-1a
digest -- the same rolling hash ``evals.suite`` already diffs against numpy on the host.
The expected digest is computed in Python at emit time and written into ``cilicon.yml``
as the string the console has to print. A firmware that boots, runs and computes anything
different fails the check. "It linked" and "it booted" are both weaker claims than the one
this makes, which is that the int8 kernel is bit-exact on real cross-compiled ARM.

**What the ELF is and is not.** It is linked against the *emulator's* memory map, because
that is what has to boot. It is not a flashable image for the part named in
``--device``: that part's flash and SRAM figures are used as the size *gate*
(``flash_max`` / ``ram_max``), not as the link map. Boot proves the code runs and is
correct; the gate proves it fits. Neither is a measurement on silicon, and nothing here
measures power -- see ``energy_estimate``, which is explicitly derived.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neuraltransistor.ir.graph import CircuitIR
from neuraltransistor.ir.runtime import Reference
from neuraltransistor.target.devices import DEVICES
from neuraltransistor.target.mcu_int8 import emit_c


@dataclass(frozen=True)
class Machine:
    """An emulated Cortex-M target: what to build for and what to boot on."""
    name: str
    mcpu: str
    qemu_machine: str
    flash_origin: int
    flash_len: int
    ram_origin: int
    ram_len: int
    note: str = ""


#: Emulated parts cilicon can boot. The memory maps are the emulator's, not a real
#: board's -- see the module docstring on what the ELF is and is not.
MACHINES: dict[str, Machine] = {m.name: m for m in [
    Machine("lm3s6965evb", "cortex-m3", "lm3s6965evb",
            0x00000000, 256 * 1024, 0x20000000, 64 * 1024,
            "Cortex-M3. Small RAM: circuits over ~60 KB of state will not link here."),
    Machine("mps2-an385", "cortex-m3", "mps2-an385",
            0x00000000, 4 * 1024 * 1024, 0x20000000, 4 * 1024 * 1024,
            "Cortex-M3 with room. The default: big enough that a link failure means "
            "the circuit is genuinely too big, not that the emulator is small."),
    Machine("mps2-an500", "cortex-m7", "mps2-an500",
            0x00000000, 8 * 1024 * 1024, 0x20000000, 8 * 1024 * 1024,
            "Cortex-M7, matching the STM32H743 core."),
    Machine("mps2-an521", "cortex-m33", "mps2-an521",
            0x00000000, 4 * 1024 * 1024, 0x20000000, 4 * 1024 * 1024,
            "Cortex-M33, matching STM32U575 and RP2350."),
    Machine("mps3-an547", "cortex-m55", "mps3-an547",
            0x00000000, 8 * 1024 * 1024, 0x20000000, 8 * 1024 * 1024,
            "Cortex-M55 with Helium, matching Apollo510."),
]}

#: Which emulated core matches which real part, for `--device`.
DEVICE_MACHINE = {
    "STM32F405": "mps2-an385", "STM32F401": "mps2-an385", "nRF52840": "mps2-an385",
    "STM32H743": "mps2-an500",
    "STM32U575": "mps2-an521", "RP2350": "mps2-an521",
    "Apollo510": "mps3-an547", "Apollo4Plus": "mps2-an521",
    "ESP32-S3": "mps2-an385",   # Xtensa, not ARM: the core does not match, only the size gate does
}


@dataclass
class FirmwareReport:
    circuit: str
    machine: str
    device: str | None
    ticks: int
    expect: str
    files: list = field(default_factory=list)
    text_b: int | None = None
    data_b: int | None = None
    bss_b: int | None = None
    #: The emitter's arithmetic estimate for the kernel alone, for comparison.
    kernel_ram_estimate_b: int | None = None
    #: Input buffer the CALLER supplies: one int16 of drive per neuron. Not part of the
    #: kernel's own footprint, which is why the estimate never counted it -- and 18% of
    #: total RAM on compass, which is why the headline number was wrong to omit it.
    input_buffer_b: int | None = None
    built: bool = False
    build_error: str = ""

    @property
    def flash_b(self) -> int | None:
        """What lands in flash: .text + .rodata + the .data initialisers."""
        if self.text_b is None:
            return None
        return self.text_b + (self.data_b or 0)

    @property
    def ram_b(self) -> int | None:
        if self.bss_b is None:
            return None
        return self.bss_b + (self.data_b or 0)

    def __str__(self) -> str:
        if not self.built:
            return f"firmware[{self.circuit}] not built: {self.build_error or 'not attempted'}"
        return (f"firmware[{self.circuit}] {self.machine}: "
                f"flash {self.flash_b / 1024:.1f} KB (.text {self.text_b:,} "
                f"+ .data {self.data_b:,}), ram {self.ram_b / 1024:.1f} KB "
                f"(.bss {self.bss_b:,} + .data {self.data_b:,})")


# --------------------------------------------------------------------------- #
# the expected digest, computed against the numpy reference
# --------------------------------------------------------------------------- #

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = 0xFFFFFFFFFFFFFFFF


def _drive_pattern(n: int) -> np.ndarray:
    """The fixed input the firmware runs against. Matches ``evals.suite``."""
    return np.array([((i * 37) % 400) - 100 for i in range(n)], dtype=np.int64)


def reference_digest(ir: CircuitIR, ticks: int = 64, weight_bits: int = 8) -> str:
    """Fold ``ticks`` ticks of the numpy reference into one FNV-1a digest.

    This is the string the firmware has to print. Computing it here rather than
    reading it back off the console is the whole point: the expectation is derived
    from the reference implementation, so the console can only match it by computing
    the same spikes.
    """
    ref = Reference(ir, weight_bits=weight_bits)
    drive = _drive_pattern(ir.n_neurons)
    h = _FNV_OFFSET
    for _ in range(ticks):
        fired = ref.tick(drive)
        for b in fired:
            h = ((h ^ int(b)) * _FNV_PRIME) & _MASK
    return f"{h:016x}"


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #

_STARTUP_C = r"""/* generated by neuraltransistor -- startup for "%(p)s"
 *
 * Reset entry for Cortex-M. The vector table's first word is the initial stack
 * pointer and the second is the reset vector; the core loads both in hardware
 * before executing anything, which is why this table must be at the start of flash.
 *
 * .data is copied out of flash into SRAM and .bss is zeroed here. Skipping that is
 * the classic bare-metal bug that works anyway in an emulator -- which starts RAM
 * zeroed -- and then fails on silicon, which does not.
 */
#include <stdint.h>

extern int main(void);

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _stack_top;

void reset_handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss; ) *dst++ = 0u;
    main();
    for (;;) { }
}

static void hang(void) { for (;;) { } }

__attribute__((section(".vectors"), used))
void (*const vectors[])(void) = {
    (void (*)(void))(&_stack_top),   /* initial SP  */
    reset_handler,                   /* reset       */
    hang,                            /* NMI         */
    hang,                            /* HardFault   */
    hang, hang, hang, 0, 0, 0, 0,    /* MM/Bus/Usage, reserved */
    hang,                            /* SVC         */
    0, 0,
    hang,                            /* PendSV      */
    hang,                            /* SysTick     */
};
"""

_MAIN_C = r"""/* generated by neuraltransistor -- boot harness for "%(p)s"
 *
 * Runs the circuit for %(ticks)d ticks against a fixed input and folds every spike
 * into an FNV-1a digest, then prints it. The expected digest was computed from the
 * numpy reference at emit time, so a matching line proves this cross-compiled ARM
 * build computed bit-identical spikes -- not merely that it booted.
 */
#include <stdint.h>
#include "%(p)s.h"

/* ARM semihosting: the emulator's console. bkpt 0xAB traps to the host. */
static int semihost(int op, void *arg) {
    register int r0 __asm__("r0") = op;
    register void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}
static void puts_(const char *s) { semihost(0x04 /* SYS_WRITE0 */, (void *)s); }

static void put_hex64(uint64_t v, char *out) {
    static const char D[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) out[15 - i] = D[(v >> (4 * i)) & 0xF];
    out[16] = 0;
}

static %(p)s_state_t st;
static int16_t drive[%(P)s_N_NEURONS];

int main(void) {
    puts_("neuraltransistor %(p)s\n");

    for (int i = 0; i < %(P)s_N_NEURONS; ++i)
        drive[i] = (int16_t)(((i * 37) %% 400) - 100);

    %(p)s_reset(&st);

    uint64_t h = 0xCBF29CE484222325ULL;
    uint32_t spikes = 0;
    for (int t = 0; t < %(ticks)d; ++t) {
        %(p)s_tick(&st, drive);
        for (int i = 0; i < %(P)s_N_NEURONS; ++i) {
            h ^= (uint64_t)st.fired[i];
            h *= 0x100000001B3ULL;
            spikes += st.fired[i];
        }
    }

    char buf[24];
    put_hex64(h, buf);
    puts_("DIGEST ");
    puts_(buf);
    puts_("\n");

    put_hex64((uint64_t)spikes, buf);
    puts_("SPIKES ");
    puts_(buf + 8);          /* spikes fit comfortably in 32 bits */
    puts_("\n");

    /* SYS_EXIT(ApplicationExit) so the emulator returns instead of spinning */
    int args[2] = {0x20026, 0};
    semihost(0x18, args);
    for (;;) { }
    return 0;
}
"""

_LD = r"""/* generated by neuraltransistor -- link map for %(machine)s (%(mcpu)s)
 *
 * This is the EMULATOR's memory map, not %(device_note)s. It exists so the image
 * boots under qemu-system-arm; fitting a real part is checked separately, by
 * comparing the linked section sizes against that part's datasheet capacity.
 */
MEMORY {
  FLASH (rx) : ORIGIN = 0x%(flash_origin)08X, LENGTH = %(flash_len)d
  SRAM  (rw) : ORIGIN = 0x%(ram_origin)08X, LENGTH = %(ram_len)d
}
ENTRY(reset_handler)
SECTIONS {
  .text : {
    KEEP(*(.vectors))
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
  } > FLASH
  _sidata = LOADADDR(.data);
  .data : {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > SRAM AT > FLASH
  .bss (NOLOAD) : {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > SRAM
  _stack_top = ORIGIN(SRAM) + LENGTH(SRAM);
}
"""

_MAKEFILE = r"""# generated by neuraltransistor -- cross-build for %(machine)s
CROSS   ?= arm-none-eabi-
CC       = $(CROSS)gcc
SIZE     = $(CROSS)size
OBJCOPY  = $(CROSS)objcopy

CFLAGS  = -mcpu=%(mcpu)s -mthumb -Os -std=c99 -ffreestanding \
          -fno-common -ffunction-sections -fdata-sections \
          -Wall -Wextra -Werror -I.
LDFLAGS = -nostdlib -nostartfiles -Wl,--gc-sections -Wl,--build-id=none \
          -T %(p)s.ld -Wl,-Map=%(p)s.map

SRC = main.c startup.c %(p)s.c %(p)s_runtime.c

all: %(p)s.elf size

%(p)s.elf: $(SRC) %(p)s.ld
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRC) -o $@

size: %(p)s.elf
	$(SIZE) $<

bin: %(p)s.elf
	$(OBJCOPY) -O binary $< %(p)s.bin

run: %(p)s.elf
	qemu-system-arm -M %(qemu_machine)s -nographic -semihosting -kernel $<

clean:
	rm -f %(p)s.elf %(p)s.bin %(p)s.map

.PHONY: all size bin run clean
"""

_CILICON = """# generated by neuraltransistor -- cross-build and boot the emitted kernel.
#
# The `expect` string is the FNV-1a digest of %(ticks)d ticks of spikes, computed from
# the numpy reference at emit time. The console can only print it by computing the same
# spikes, so a green check means the int8 kernel is bit-exact on cross-compiled ARM --
# not merely that the firmware linked and reached main.
#
# flash_max / ram_max are %(device)s's datasheet capacity. The ELF is linked against the
# EMULATOR's map (see %(p)s.ld); these gates are what check it would fit the real part.
targets:
  - id: %(p)s/%(machine)s
    base: debian:bookworm-slim
    apt: [gcc-arm-none-eabi, qemu-system-arm]
    build: make -C %(dir)s
    validate: qemu_system
    machine: %(qemu_machine)s
    artifact: %(dir)s/%(p)s.elf
    expect: "DIGEST %(expect)s"
    size_tool: arm-none-eabi-size
    flash_max: %(flash_max)s
    ram_max: %(ram_max)s
    boot_timeout: 120
    artifacts: [%(dir)s/%(p)s.elf]
"""


def _kernel_ram_estimate(ir: CircuitIR) -> int:
    """What the emitter predicts for the kernel alone: state struct + accumulator."""
    n = ir.n_neurons
    return n * (2 + 1 + 1 + (1 if ir.n_mod_edges else 0)) + n * 4


def _fmt_kb(n: int) -> str:
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)


def emit_firmware(ir: CircuitIR, outdir: str | Path, device: str | None = None,
                  machine: str | None = None, ticks: int = 64,
                  weight_bits: int = 8, build: bool = True,
                  prefix: str | None = None) -> FirmwareReport:
    """Write a complete bootable project for ``ir`` into ``outdir``.

    Returns a :class:`FirmwareReport` carrying the linker's section sizes when
    ``build`` is set and ``arm-none-eabi-gcc`` is on PATH.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = prefix or ir.name.replace("-", "_")

    if machine is None:
        machine = DEVICE_MACHINE.get(device or "", "mps2-an385")
    if machine not in MACHINES:
        raise ValueError(f"unknown machine {machine!r}; have {sorted(MACHINES)}")
    m = MACHINES[machine]

    emit_c(ir, outdir, weight_bits=weight_bits, prefix=p)
    expect = reference_digest(ir, ticks=ticks, weight_bits=weight_bits)

    sub = {"p": p, "P": p.upper(), "ticks": ticks, "machine": m.name,
           "mcpu": m.mcpu, "qemu_machine": m.qemu_machine,
           "flash_origin": m.flash_origin, "flash_len": m.flash_len,
           "ram_origin": m.ram_origin, "ram_len": m.ram_len,
           "device_note": (f"{device}'s" if device else "any real part's"),
           "expect": expect, "device": device or "the selected part",
           "dir": outdir.as_posix()}

    dev = DEVICES.get(device) if device else None
    sub["flash_max"] = _fmt_kb(dev.flash_b) if dev else _fmt_kb(m.flash_len)
    sub["ram_max"] = _fmt_kb(dev.sram_b) if dev else _fmt_kb(m.ram_len)

    written = []
    for name, body in [(f"{p}.ld", _LD), ("startup.c", _STARTUP_C),
                       ("main.c", _MAIN_C), ("Makefile", _MAKEFILE),
                       ("cilicon.yml", _CILICON)]:
        (outdir / name).write_text(body % sub)
        written.append(str(outdir / name))

    rep = FirmwareReport(circuit=ir.name, machine=m.name, device=device,
                         ticks=ticks, expect=expect, files=written)
    rep.input_buffer_b = ir.n_neurons * 2
    rep.kernel_ram_estimate_b = _kernel_ram_estimate(ir)
    if build:
        _build(outdir, rep)
    return rep


def _build(outdir: Path, rep: FirmwareReport) -> FirmwareReport:
    make = shutil.which("make")
    cc = shutil.which("arm-none-eabi-gcc")
    if cc is None or make is None:
        rep.build_error = ("arm-none-eabi-gcc not on PATH"
                           if cc is None else "make not on PATH")
        return rep
    cp = subprocess.run([make, "-C", str(outdir)], capture_output=True, text=True)
    if cp.returncode != 0:
        rep.build_error = (cp.stderr or cp.stdout)[-2000:]
        return rep
    rep.built = True
    size = shutil.which("arm-none-eabi-size")
    if size:
        out = subprocess.run([size, str(outdir / f"{rep.circuit.replace('-', '_')}.elf")],
                             capture_output=True, text=True)
        rows = [l.split() for l in out.stdout.strip().splitlines()[1:] if l.strip()]
        if rows and len(rows[0]) >= 3:
            rep.text_b, rep.data_b, rep.bss_b = (int(rows[0][0]), int(rows[0][1]),
                                                 int(rows[0][2]))
    return rep


# --------------------------------------------------------------------------- #
# energy -- derived, and labelled as such
# --------------------------------------------------------------------------- #

def energy_estimate(device: str, ticks_per_second: float,
                    cycles_per_tick: float | None = None,
                    voltage: float = 1.8) -> dict:
    """Energy per tick, DERIVED from datasheet current draw. Not a measurement.

    There is no power figure measured on silicon anywhere in this repo, and this
    function does not produce one. It multiplies a datasheet number by an arithmetic
    duty cycle, which is the same class of claim as the flash fits -- useful for ruling
    options out, worthless as evidence that a robot's battery lasts.

    A real number needs a shunt on a real board: ``ina219`` / ``ina226`` are the parts,
    and cilicon's catalog lists both. Until one of those is in the loop, every field
    below is an estimate and is named so.
    """
    d = DEVICES.get(device)
    if d is None:
        raise KeyError(f"unknown device {device!r}")
    if d.active_mw is None:
        return {"device": device, "estimate": None,
                "why": "no datasheet active power for this part"}
    # Datasheet active power is quoted at the part's nominal rail; scaling to another
    # voltage assumes dynamic power dominates and goes as V^2, which is only roughly
    # true and is wrong in the leakage-dominated regime this class of part often sits in.
    nominal_v = 3.3
    scale = (voltage / nominal_v) ** 2
    active_mw = d.active_mw * scale
    duty = None
    energy_uj = None
    if cycles_per_tick:
        seconds_per_tick = cycles_per_tick / (d.clock_mhz * 1e6)
        duty = min(seconds_per_tick * ticks_per_second, 1.0)
        energy_uj = active_mw * 1e3 * seconds_per_tick / 1e6
    return {
        "device": device,
        "measured": False,
        "basis": f"datasheet active power {d.active_mw} mW at {nominal_v} V ({d.source})",
        "voltage_v": voltage,
        "active_mw_at_voltage": round(active_mw, 3),
        "voltage_scaling": "V^2, dynamic-power approximation only",
        "cycles_per_tick": cycles_per_tick,
        "ticks_per_second": ticks_per_second,
        "duty_cycle": duty,
        "energy_per_tick_uj": energy_uj,
        "mean_mw": round(active_mw * duty, 4) if duty is not None else None,
        "caveat": "derived from datasheet arithmetic; nothing here was measured on silicon",
    }
