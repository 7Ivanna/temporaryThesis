import numpy as np
from matplotlib import pyplot as plt

def gaussian_pulse(E, t, t0, tau_p):
    phi = E * np.exp(-4 * np.log(2) * (((t - t0) / tau_p) ** 2)) # got this from RP photonics , kind of 
    return phi

def p_arb_gaussian_pulse(E, t1, t2, t10, t20, tau_p, t):
    output = np.zeros(len(t), dtype=np.complex128)
    for i in range(len(t)):
        part1 = np.trapz(np.conj(gaussian_pulse(E, t1, t10, tau_p)) * gaussian_pulse(E, t1, t20, tau_p) * (np.cos(t1 * t[i]) - 1j * np.sin(t1 * t[i])), t1)
        part2 = np.trapz(np.conj(gaussian_pulse(E, t2, t20, tau_p)) * gaussian_pulse(E, t2, t10, tau_p) * (np.cos(t2 * t[i]) + 1j * np.sin(t2 * t[i])), t2)
        output[i] = 1 / 2 - 1 / 2 * (part1 * part2)
    return output

def fft_freqs(t):
    N = len(t)
    dt = t[1] - t[0]
    freqs = np.fft.fftfreq(N, dt)
    return np.fft.fftshift(freqs)

def hom_measurement(t, t10, t20):
    tau_p = 2.355*(10**(-9)) # FWHM of my gaussian pulse 
    t1 = fft_freqs(t) * 2 * np.pi
    t2 = fft_freqs(t) * 2 * np.pi
    E = 1
    return np.abs(p_arb_gaussian_pulse(E, t1, t2, t10, t20, tau_p, t))

tau_p = 2.355*(10**(-9))
t = np.linspace(-5, 5, 1000)
t10 = 0.8*tau_p
t20 = 0.8*tau_p

Hom = hom_measurement(t, t10, t20)

plt.plot(t, Hom)
plt.xlabel('Time (tau=10^-9)')
plt.ylabel('Coincidence Probability ')
plt.title('Hong-Ou-Mandel Interference')
plt.show()
