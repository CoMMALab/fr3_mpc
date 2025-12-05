import numpy as np


def low_freq_sine(t: float, joint: int, amp: float = 15.0) -> float:
    """Low-frequency sine for gravity + friction identification."""
    f = 0.3  # Hz
    phase = joint * 0.7
    return 0.5 * amp * np.sin(2 * np.pi * f * t + phase)

def multisine(t: float, joint: int, amp: float = 15.0) -> float:
    """Sum of multiple frequencies for broad-spectrum excitation."""
    freqs = [0.2, 0.5, 1.0, 2.0, 3.5, 5.0]
    tau = 0.0
    amp_per_freq = amp / len(freqs)
    for k, f in enumerate(freqs):
        phase = (joint * 1.7 + k * 0.3) * 2 * np.pi
        tau += amp_per_freq * np.sin(2 * np.pi * f * t + phase)
    return tau

def chirp(t: float, amp: float = 15.0, T: float = 15.0) -> float:
    """Frequency sweep from 0.1 Hz to 8 Hz."""
    f0, f1 = 0.1, 8.0
    phase_t = 2*np.pi * (f0*t + 0.5*(f1-f0)*t**2/T)
    return 0.4 * amp * np.sin(phase_t)

def imp(t: float, length: float, amp: float) -> float:
    """Single rectangular impulse starting at t=0, lasting <length> seconds."""
    return amp if 0 <= t <= length else 0.0

def lin(t: float, m: float, b: float) -> float:
    """Linear function m*t + b."""
    return m*t + b

def sin(t: float, f: float, a: float) -> float:
    """Sine wave with frequency f (Hz) and amplitude a."""
    return a * np.sin(2*np.pi*f*t)

def square(t: float, f: float, a: float) -> float:
    """Square wave with frequency f and amplitude a."""
    return a * np.sign(np.sin(2*np.pi*f*t))

def apply_to_joint(signal_fn, t: float, joint: int, **kwargs) -> np.ndarray:
    vec = np.zeros(7)
    vec[joint] = signal_fn(t, **kwargs)
    return vec