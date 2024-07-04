import numpy as np
from matplotlib import pyplot as plt

def gaussian_pulse_freq(w, w2, sigma):
    return (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

def p_arb_gaussian_pulse(w_1, w_2, w10, w20, sigma, tau):
    total = np.zeros(len(tau), dtype=np.complex128)
    for i in range(len(tau)):
        part1 = np.trapz(np.conj(gaussian_pulse_freq(w_1, w10, sigma)) * gaussian_pulse_freq(w_1, w20, sigma) * (np.cos(w_1 * tau[i]) - 1j * np.sin(w_1 * tau[i])), w_1)
        part2 = np.trapz(np.conj(gaussian_pulse_freq(w_2, w20, sigma)) * gaussian_pulse_freq(w_2, w10, sigma) * (np.cos(w_2 * tau[i]) + 1j * np.sin(w_2 * tau[i])), w_2)
        total[i] = 1 / 2 - (1 / 2 * (part1 * part2))
    return total

def fft_freqs(t):
    w = np.zeros(len(t))
    for i in range(len(t)):
        w[i] = (i - len(t) // 2) / ((t[1] - t[0]) * len(t))
    return w

def hom_measurement(w10, w20, sigma):
    tau = np.linspace(-10/sigma, 10/sigma, 300)
    w1 = fft_freqs(tau) * 2 * np.pi
    return np.abs(p_arb_gaussian_pulse(w1, w1, w10, w20, sigma, tau))

# Parameters
w10 = 0
w20 = 0
pulse_width_ns = 1e-9 # 1 ns pulse width
sigma_t = pulse_width_ns / (2 * np.sqrt(2 * np.log(2))) # converted sigma 
sigma_f = 1 / (2 * np.pi * sigma_t)  # sigma in freque3ncy

# Print parameters to check values
print("w10:", w10)
print("w20:", w20)
print("sigma_t (ns):", sigma_t)
print("sigma_f (GHz):", sigma_f)

hom_result = hom_measurement(w10, w20, sigma_f)

tau = np.linspace(-10/sigma_f, 10/sigma_f, 300)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot( tau,hom_result)
plt.title('HOM Measurement using Gaussian Pulse')
plt.xlabel('Time [s]')
plt.ylabel('Coincidence Amplitude')
plt.grid(True)
plt.show()



# %%
import numpy as np
from matplotlib import pyplot as plt

def gaussian_pulse_freq(w, w2, sigma):
    return (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

def p_arb_gaussian_pulse(w_1, w_2, w10, w20, sigma, tau, time_jitter):
    total = np.zeros(len(tau), dtype=np.complex128)
    jitter = np.random.normal(0, time_jitter, len(tau))  # Adding time jitter
    for i in range(len(tau)):
        t_jittered = tau[i] + jitter[i]
        part1 = np.trapz(np.conj(gaussian_pulse_freq(w_1, w10, sigma)) * gaussian_pulse_freq(w_1, w20, sigma) * np.exp(-1j * w_1 * t_jittered), w_1)
        part2 = np.trapz(np.conj(gaussian_pulse_freq(w_2, w20, sigma)) * gaussian_pulse_freq(w_2, w10, sigma) * np.exp(1j * w_2 * t_jittered), w_2)
        total[i] = 1 / 2 - (1 / 2 * (part1 * part2))
    return total

def fft_freqs(t):
    w = np.zeros(len(t))
    for i in range(len(t)):
        w[i] = (i - len(t) // 2) / ((t[1] - t[0]) * len(t))
    return w

def hom_measurement(w10, w20, sigma, sigma_jitter):
    tau = np.linspace(-10 / sigma, 10 / sigma, 1000)
    w1 = fft_freqs(tau) * 2 * np.pi
    return np.abs(p_arb_gaussian_pulse(w1, w1, w10, w20, sigma, tau, sigma_jitter))

# Parameters
w10 = 0

pulse_width_ns = 1e-9  # 1 ns pulse width
sigma_t = pulse_width_ns / (2 * np.sqrt(2 * np.log(2)))  # converted sigma 
sigma_f = 1/ (2 * np.pi * sigma_t)  # sigma in frequency
time_jitter = 0.5 * pulse_width_ns  # Time jitter standard deviation
w20 = (time_jitter )

hom_result = hom_measurement(w10, w20, sigma_f, time_jitter)

tau = np.linspace(-10 / sigma_f, 10 / sigma_f, 1000)
plt.figure(figsize=(10, 6))
plt.plot(tau, hom_result,label='0.5')
plt.legend()
plt.title('HOM Measurement using Gaussian Pulse with Time Jitter')
plt.xlabel('Time [s]')
plt.ylabel('Coincidence Amplitude')
plt.grid(True)
plt.show()

# %%
import numpy as np
from matplotlib import pyplot as plt

# Define the Gaussian pulse in the frequency domain
def gaussian_pulse_freq(w, w2, sigma):
    return (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

# Define the HOM measurement with integrals over frequency
def p_arb_gaussian_pulse(w_1, w_2, w10, w20, sigma, tau):
    total = np.zeros(len(tau), dtype=np.complex128)
    for i in range(len(tau)):
        t = tau[i]
        part1 = np.trapz(np.conj(gaussian_pulse_freq(w_1, w10, sigma)) * gaussian_pulse_freq(w_1, w20, sigma) * np.exp(-1j * w_1 * t), w_1)
        part2 = np.trapz(np.conj(gaussian_pulse_freq(w_2, w20, sigma)) * gaussian_pulse_freq(w_2, w10, sigma) * np.exp(1j * w_2 * t), w_2)
        total[i] = 1 / 2 - (1 / 2 * (part1 * part2))
    return total

# Function to compute FFT frequencies
def fft_freqs(t):
    w = np.fft.fftfreq(len(t), d=t[1] - t[0])
    return np.fft.fftshift(w) * 2 * np.pi

# Function to perform the HOM measurement
def hom_measurement(w10, w20, sigma):
    tau = np.linspace(-10 / sigma, 10 / sigma, 2000)  # Increased resolution
    w1 = fft_freqs(tau)
    return np.abs(p_arb_gaussian_pulse(w1, w1, w10, w20, sigma, tau))

# Parameters
pulse_width_ns = 1e-9  # 1 ns pulse width

w10 = 0

sigma_t = pulse_width_ns / (2 * np.sqrt(2 * np.log(2)))  # converted sigma in time domain
sigma_f = 1 / (2 * np.pi * sigma_t)  # sigma in frequency domain
w20 =  (1/pulse_width_ns)*0.5 # Center frequency of pulse 2

# Perform the HOM measurement
hom_result = hom_measurement(w10, w20, sigma_f)

# Plot the HOM interference
tau = np.linspace(-10 / sigma_f, 10 / sigma_f, 2000)  # Increased resolution
plt.figure(figsize=(10, 6))
plt.plot(tau, hom_result, label='HOM Interference')
plt.legend()
plt.title('HOM Measurement using Gaussian Pulse without Time Jitter')
plt.xlabel('Time [s]')
plt.ylabel('Coincidence Amplitude')
plt.grid(True)
plt.show()

# %%
import numpy as np
from matplotlib import pyplot as plt

def gaussian_pulse_freq(w, w2, sigma):
    return (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

# Parameters
pulse_width_ns = 1e-9  # 1 ns pulse width

w10 = 0

sigma_t = pulse_width_ns / (2 * np.sqrt(2 * np.log(2)))  # converted sigma in time domain
sigma_f = 1 / (2 * np.pi * sigma_t)  # sigma in frequency domain
w20 =  (1/pulse_width_ns)*0.5 # Center frequency of pulse 2
tau = np.linspace(-10 / sigma_f, 10 / sigma_f, 2000)  # Increased resolution

# Perform the HOM measurement
hom_result = gaussian_pulse_freq(tau, w20, sigma_f)

# Plot the HOM interference
plt.figure(figsize=(10, 6))
plt.plot(tau, hom_result, label='HOM Interference')
plt.legend()
plt.title('HOM Measurement using Gaussian Pulse without Time Jitter')
plt.xlabel('Time [s]')
plt.ylabel('Coincidence Amplitude')
plt.grid(True)
plt.show()




# %%
# its time to do the MC sims yay 

import numpy as np
from matplotlib import pyplot as plt
def mc_sim(tj,):
    # Define the Gaussian pulse in the frequency domain
    def gaussian_pulse_freq(w, w2, sigma):
        return (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

    # Define the HOM measurement with integrals over frequency
    def p_arb_gaussian_pulse(w_1, w_2, w10, w20, sigma, tau):
        total = np.zeros(len(tau), dtype=np.complex128)
        for i in range(len(tau)):
            t = tau[i]
            part1 = np.trapz(np.conj(gaussian_pulse_freq(w_1, w10, sigma)) * gaussian_pulse_freq(w_1, w20, sigma) * np.exp(-1j * w_1 * t), w_1)
            part2 = np.trapz(np.conj(gaussian_pulse_freq(w_2, w20, sigma)) * gaussian_pulse_freq(w_2, w10, sigma) * np.exp(1j * w_2 * t), w_2)
            total[i] = 1 / 2 - (1 / 2 * (part1 * part2))
        return total

    # Function to compute FFT frequencies
    def fft_freqs(t):
        w = np.fft.fftfreq(len(t), d=t[1] - t[0])
        return np.fft.fftshift(w) * 2 * np.pi

    # Function to perform the HOM measurement
    def hom_measurement(w10, w20, sigma):
        tau = np.linspace(-10 / sigma, 10 / sigma, 2000)  # Increased resolution
        w1 = fft_freqs(tau)
        return np.abs(p_arb_gaussian_pulse(w1, w1, w10, w20, sigma, tau))

    # Parameters
    pulse_width_ns = 1e-9  # 1 ns pulse width

    w10 = 0

    sigma_t = pulse_width_ns / (2 * np.sqrt(2 * np.log(2)))  # converted sigma in time domain
    sigma_f = 1 / (2 * np.pi * sigma_t)  # sigma in frequency domain
    w20 =  tj

    # Perform the HOM measurement
    hom_result = hom_measurement(w10, w20, sigma_f)

    # Plot the HOM interference
    tau = np.linspace(-10 / sigma_f, 10 / sigma_f, 2000)  # Increased resolution
    return 
plt.figure(figsize=(10, 6))
plt.plot(tau, hom_result, label='HOM Interference')
plt.legend()
plt.title('HOM Measurement using Gaussian Pulse without Time Jitter')
plt.xlabel('Time [s]')
plt.ylabel('Coincidence Amplitude')
plt.grid(True)
plt.show()
# initialize variables
n_simulations = 100
# create lists to store x and y values
l_xs = [] # 
l_ys = [] # time 
time_jitter = (0.5/pulse_width_ns) # Center frequency of pulse 2

# loop n_simulations times
for i in range(n_simulations):
    pulse_width_ns = 1e-9  # 1 ns pulse width
    time_jitter = (1/pulse_width_ns)*0.5 # Center frequency of pulse 2
    # x is randomly drawn from a continuous uniform distritbuion
    x = np.random.uniform(-time_jitter, time_jitter)
    # store x in the list
    l_xs.append(x)
    
    # y is randomly drawn from a continuous uniform distribution
    y = np.random.uniform(-1, 1)
    # store y in the list
    l_ys.append(y)
    mc_sim(x,y)


# %%
import numpy as np
from matplotlib import pyplot as plt

def mc_sim(tj, pulse_width_ns):
    # Define the Gaussian pulse in the frequency domain
    def gaussian_pulse_freq(w, w2, sigma):
        return (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

    # Define the HOM measurement with integrals over frequency
    def p_arb_gaussian_pulse(w_1, w_2, w10, w20, sigma, tau):
        total = np.zeros(len(tau), dtype=np.complex128)
        for i in range(len(tau)):
            t = tau[i]
            part1 = np.trapz(np.conj(gaussian_pulse_freq(w_1, w10, sigma)) * gaussian_pulse_freq(w_1, w20, sigma) * np.exp(-1j * w_1 * t), w_1)
            part2 = np.trapz(np.conj(gaussian_pulse_freq(w_2, w20, sigma)) * gaussian_pulse_freq(w_2, w10, sigma) * np.exp(1j * w_2 * t), w_2)
            total[i] = 1 / 2 - (1 / 2 * (part1 * part2))
        return total

    # Function to compute FFT frequencies
    def fft_freqs(t):
        w = np.fft.fftfreq(len(t), d=t[1] - t[0])
        return np.fft.fftshift(w) * 2 * np.pi

    # Function to perform the HOM measurement
    def hom_measurement(w10, w20, sigma):
        tau = np.linspace(-10 / sigma, 10 / sigma, 1000)  # Increased resolution
        w1 = fft_freqs(tau)
        return tau, np.abs(p_arb_gaussian_pulse(w1, w1, w10, w20, sigma, tau))

    # Parameters
    sigma_t = pulse_width_ns / (2 * np.sqrt(2 * np.log(2)))  # converted sigma in time domain
    sigma_f = 1 / (2 * np.pi * sigma_t)  # sigma in frequency domain

    # Perform the HOM measurement
    return hom_measurement(0, tj, sigma_f)

# Monte Carlo simulation parameters
n_simulations = 1000
pulse_width_ns = 1  # 1 ns pulse width

# List of different time jitter values to test
time_jitter_values = [0, 0.1, 0.5, .20,.9,2]  # Example time jitter values in ns
plt.figure(figsize=(10, 6))

# Run Monte Carlo simulations for each time jitter value
for time_jitter_ns in time_jitter_values:
    time_jitter_range = time_jitter_ns / pulse_width_ns  # freq
  

    # Arrays to accumulate results
    tau_accum = None
    hom_accum = None

    for _ in range(n_simulations):
        time_jitter = np.random.normal(0, time_jitter_range, 1)
        tau, hom_result = mc_sim(time_jitter, pulse_width_ns)
        if tau_accum is None:
            tau_accum = tau
            hom_accum = hom_result
        else:
            hom_accum += hom_result

    # Average the results
    hom_avg = hom_accum / n_simulations

    # Plot the averaged HOM interference for this time jitter value
    plt.plot(tau_accum, hom_avg, label=f'Time Jitter = {time_jitter_ns} ns')

plt.legend()
plt.title('HOM Measurement using Gaussian Pulse with Various Time Jitters (Monte Carlo Simulation)')
plt.xlabel('Time [ns]')
plt.ylabel('Coincidence Amplitude')
plt.grid(True)
plt.show()

# %%
#this one does many
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def mc_sim(tj, pulse_width_ns):
    # Define the Gaussian pulse in the frequency domain
    def gaussian_pulse_freq(w, w2, sigma):
        return (1 / np.sqrt(np.sqrt(np.pi) * sigma)) * np.exp(-((w - w2) ** 2) / (2 * sigma ** 2))

    # Define the HOM measurement with integrals over frequency
    def p_arb_gaussian_pulse(w_1, w_2, w10, w20, sigma, tau):
        total = np.zeros(len(tau), dtype=np.complex128)
        for i in range(len(tau)):
            t = tau[i]
            part1 = np.trapz(np.conj(gaussian_pulse_freq(w_1, w10, sigma)) * gaussian_pulse_freq(w_1, w20, sigma) * np.exp(-1j * w_1 * t), w_1)
            part2 = np.trapz(np.conj(gaussian_pulse_freq(w_2, w20, sigma)) * gaussian_pulse_freq(w_2, w10, sigma) * np.exp(1j * w_2 * t), w_2)
            total[i] = 1 / 2 - (1 / 2 * (part1 * part2))
        return total

    # Function to compute FFT frequencies
    def fft_freqs(t):
        w = np.fft.fftfreq(len(t), d=t[1] - t[0])
        pr= np.fft.fftshift(w) * 2 * np.pi
        return pr

    # Function to perform the HOM measurement
    def hom_measurement(w10, w20, sigma):
        tau = np.linspace(-10 / sigma, 10 / sigma, 1000)  # Increased resolution
        w1 = fft_freqs(tau)
        return w1, tau, np.abs(p_arb_gaussian_pulse(w1, w1, w10, w20, sigma, tau))
    
    # Parameters
    sigma_t = pulse_width_ns / (2 * np.sqrt(2 * np.log(2)))  # converted sigma in time domain
    sigma_f = 1 / (2 * np.pi * sigma_t)  # sigma in frequency domain

    # Perform the HOM measurement
    w1, tau, hom_result = hom_measurement(0, tj, sigma_f)

    # Define the Gaussian pulse in the time domain
    def gaussian_pulse_time(t, sigma_t):
        return (1 / (sigma_t * np.sqrt(2 * np.pi))) * np.exp(-t**2 / (2 * sigma_t**2))

    # Plot the Gaussian pulses in time and frequency domains
    t = np.linspace(-5 * pulse_width_ns, 5 * pulse_width_ns, 1000)
    pulse_time = gaussian_pulse_time(t, sigma_t)
    pulse_freq = gaussian_pulse_freq(0, w1, sigma_f)
    return tau, hom_result

# Monte Carlo simulation parameters
n_simulations = 100
pulse_width_ns = 1 # 1 ns pulse width
time_jitter_values = [0, 0.1, 0.5, .20,.9,2]  # Example time jitter values in ns
plt.figure(figsize=(10, 6))

# Run Monte Carlo simulations for each time jitter value
for time_jitter_ns in time_jitter_values:

    time_jitter_range = time_jitter_ns / pulse_width_ns  # freq
    # Arrays to accumulate results
    tau_accum = None
    hom_accum = None
    # Run Monte Carlo simulations
    for _ in range(n_simulations):
        time_jitter = np.random.normal(0, time_jitter_range,1)
       # print(time_jitter)
        tau, hom_result = mc_sim(time_jitter, pulse_width_ns)
        if tau_accum is None:
            tau_accum = tau
            hom_accum = hom_result
        else:
            hom_accum += hom_result
       
        '''plt.plot(tau, hom_result, alpha=0.3)
        plt.title(f'HOM results for each simulation\nTime Jitter range = {time_jitter_range:.2f} ns')
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude')
    plt.show()'''

    # Average the results
    hom_avg = hom_accum / n_simulations

        # Plot the averaged HOM interference for this time jitter value
    plt.plot(tau_accum, hom_avg, label=f'Time Jitter = {time_jitter_ns} ns')
plt.title('Averaged HOM Measurement')
plt.xlabel('Time [ns]')
plt.ylabel('Coincidence Amplitude')
plt.legend()
plt.tight_layout()
plt.figure(figsize=(10, 6))

plt.show()

# %%
