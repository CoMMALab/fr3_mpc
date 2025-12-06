import numpy as np


def chirp(t: float, amp: float = 15.0) -> float:
    """Frequency sweep from 0.1 Hz to 8 Hz."""
    T = 5.0
    f0, f1 = 0.1, 3.0
    phase_t = 2*np.pi * (f0*t + 0.5*(f1-f0)*t**2/T)
    return amp * np.sin(phase_t)

def multisine(t: float, amp: float = 15.0) -> float:
    """Sum of multiple frequencies for broad-spectrum excitation."""
    freqs = np.array([0.2, 0.5, 1.0, 2.0, 3.5, 5.0]) / 5
    tau = 0.0
    amp_per_freq = amp / len(freqs)
    for f in freqs:
        tau += amp_per_freq * np.sin(2 * np.pi * f * t)
    return tau

def sin(t: float, amp: float, f: float = 0.2) -> float:
    """Sine wave with frequency f (Hz) and amplitude a."""
    return amp * np.sin(2*np.pi*f*t)

EXCITATION_FUNCS = {
    "chirp": chirp,
    "multisine": multisine,
    "sin": sin
}

def make_excitation(name, joint, amp):
    if name not in EXCITATION_FUNCS:
        raise ValueError(
            f"Unknown excitation function '{name}'. "
            f"Valid: {list(EXCITATION_FUNCS.keys())}"
        )

    signal_fn = EXCITATION_FUNCS[name]
    return lambda t: np.eye(7)[joint] * signal_fn(t, amp=amp)
