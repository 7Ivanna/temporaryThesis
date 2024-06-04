import numpy as np
from matplotlib import pyplot as plt


def gaussian_pulse_freq(E, w, w2, sigma):
    return E * (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

def p_arb_gaussian_pulse(E, w_1, w_2, w10, w20, sigma, tau): # integrate frequency over time 
    total = np.zeros(len(tau), dtype=np.complex128)
    for i in range(len(tau)):
        part1 = np.trapz(np.conj(gaussian_pulse_freq(E, w_1, w10, sigma)) * gaussian_pulse_freq(E, w_1, w20, sigma) * (np.cos(w_1 * tau[i]) - 1j * np.sin(w_1 * tau[i])), w_1)
        part2 = np.trapz(np.conj(gaussian_pulse_freq(E, w_2, w20, sigma)) * gaussian_pulse_freq(E, w_2, w10, sigma) * (np.cos(w_2 * tau[i]) + 1j * np.sin(w_2 * tau[i])), w_2)
        total[i] = 1 / 2 - (1 / 2 * (part1 * part2))
    return total
def fft_freqs(t):
    w = np.zeros(len(t))
    for i in range(len(t)):
        w[i] = (i-len(t)//2)/((t[1]-t[0])*len(t))
    return w

def hom_measurement(w10, w20, E ,sigma):
    tau = np.linspace(-5*scale, 5*scale, 1000)
    w1 = fft_freqs(tau) * 2 * np.pi
   # tau = [0]
    return np.abs(p_arb_gaussian_pulse(E, w1, w1, w10, w20, sigma, tau))

# Parameters
E = 1
scale = 10e-9
sigma = 2.355* 10e-9

w10 = 0.1 * sigma
w20 = 0.1* sigma
tau = np.linspace(-5*scale, 5*scale, 1000)
w_1 = np.linspace(-10*scale, 10*scale, 1000)
w_2 = np.linspace(-10*scale, 10*scale, 1000)


P = hom_measurement(w10, w20, E ,sigma)

plt.plot(tau*scale, np.abs(P))
plt.xlabel('Time')
plt.ylabel('Coincidence Probability')
plt.title('Hong-Ou-Mandel Interference')



plt.show()
#%%

import numpy as np
from matplotlib import pyplot as plt

# Define the Gaussian pulse in the frequency domain
def gaussian_pulse_freq(E, w, w2, sigma):
    return E * (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

# Define the HOM measurement with integrals over frequency using Euler's formula with time jitter
def p_arb_gaussian_pulse(E, w_1, w_2, w10, w20, sigma, tau, sigma_j):
    total = np.zeros(len(tau), dtype=np.complex128)
    jitter = np.random.normal(0, sigma_j, len(tau))
    for i in range(len(tau)):
        t_jittered = tau[i] + jitter[i]
        part1 = np.trapz(np.conj(gaussian_pulse_freq(E, w_1, w10, sigma)) * 
                         gaussian_pulse_freq(E, w_1, w20, sigma) * 
                         np.exp(1j * w_1 * t_jittered), w_1)
        part2 = np.trapz(np.conj(gaussian_pulse_freq(E, w_2, w20, sigma)) * 
                         gaussian_pulse_freq(E, w_2, w10, sigma) * 
                         np.exp(-1j * w_2 * t_jittered), w_2)
        total[i] = 1 / 2 - 1 / 2 * (part1 * part2)
    return total

# Parameters
E = 1
sigma = 1
w10 = 0.1 * sigma
w20 = 0.1 * sigma
tau = np.linspace(-5, 5, 1000)
w_1 = np.linspace(-10, 10, 1000)
w_2 = np.linspace(-10, 10, 1000)
sigma_j = 0.05  # Standard deviation of the time jitter

# Perform HOM measurement with integrals over frequency and time jitter
P = p_arb_gaussian_pulse(E, w_1, w_2, w10, w20, sigma, tau, sigma_j)

# Plot the HOM interference with integrals over frequency
plt.plot(tau, np.abs(P))
plt.xlabel('Time')
plt.ylabel('Coincidence Probability')
plt.title('Hong-Ou-Mandel Interference with Integrals over Frequency and Time Jitter')

# Set more y ticks
plt.yticks(np.linspace(0, 1, num=11))

plt.show()


# %%
import numpy as np
from matplotlib import pyplot as plt

# Define the Gaussian pulse in the frequency domain
def gaussian_pulse_freq(E, omega, omega_bar, sigma):
    return E * np.exp(-((omega - omega_bar) ** 2) / (2 * sigma ** 2))

# Define the HOM measurement with integrals over frequency using the given formula
def p_arb_gaussian_pulse(E, omega1, omega2, omega_bar_a, omega_bar_b, sigma_a, sigma_b, tau):
    integral = np.zeros(len(tau), dtype=np.complex128)
    for i in range(len(tau)):
        integrand1 = np.exp(-((omega1 - omega_bar_a) ** 2) / (2 * sigma_a ** 2)) * \
                     np.exp(-((omega1 - omega_bar_b) ** 2) / (2 * sigma_b ** 2)) * \
                     np.exp(-1j * omega1 * tau[i])
        integral1 = np.trapz(integrand1, omega1)

        integrand2 = np.exp(-((omega2 - omega_bar_a) ** 2) / (2 * sigma_a ** 2)) * \
                     np.exp(-((omega2 - omega_bar_b) ** 2) / (2 * sigma_b ** 2)) * \
                     np.exp(1j * omega2 * tau[i])
        integral2 = np.trapz(integrand2, omega2)

        integral[i] = integral1 * integral2

    return 1/2 - (1 / (2 * np.pi * sigma_a * sigma_b)) * integral

# Parameters
E = 1
scale = 10e-9
sigma_a = 2.355 * 10e-9
sigma_b = 2.355 * 10e-9

omega_bar_a = 0.1 * sigma_a
omega_bar_b = 0.1 * sigma_b
tau = np.linspace(-5 * scale, 5 * scale, 1000)
omega1 = np.linspace(-10 * scale, 10 * scale, 1000)
omega2 = np.linspace(-10 * scale, 10 * scale, 1000)

# Perform HOM measurement with integrals over frequency
P = np.abs(p_arb_gaussian_pulse(E, omega1, omega2, omega_bar_a, omega_bar_b, sigma_a, sigma_b, tau))

# Plot the HOM interference with integrals over frequency
plt.plot(tau, P)
plt.xlabel('Time (s)')
plt.ylabel('Coincidence Probability')
plt.title('Hong-Ou-Mandel Interference')


plt.show()

# %%
