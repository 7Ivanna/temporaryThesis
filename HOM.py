
#%%
#this is the good one i am using
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def sim(time_jitter, pulse_width_ns):
    def gaussian_pulse_time(t, tj, tau, sigma_t):
            normalization_factor =1.1526339143613291
            return normalization_factor * np.exp(-((t - tj - tau) ** 2) / (2 * sigma_t**2))

    def hom_in_time(t, tj1, tj, tau, sigma_t):
        total = np.zeros(len(tau))
        for i in range(len(tau)):
            t_val = t
            # Compute the overlap integral
            part1 = np.trapz(np.conj(gaussian_pulse_time(t_val, tj1, 0, sigma_t)) * gaussian_pulse_time(t_val, tj, tau[i], sigma_t),t_val)
            part2= np.trapz(np.conj(gaussian_pulse_time(t_val, tj, tau[i], sigma_t))* gaussian_pulse_time(t_val, tj1, 0, sigma_t),t_val )
            total[i] = 1/2 - 1/2 * ((part1*part2))
            
        return tau,total

    sigma_t = 1 / (2 * np.sqrt(2 * np.log(2))) # converted sigma 
   

    t = np.linspace(-10, 10, 1000)
    
    tau = np.linspace(-10, 10, 1000)
    
    return hom_in_time(t, 0, time_jitter, tau, sigma_t)

# Monte Carlo simulation parameters
n_simulations = 10000
pulse_width_ns = 1  # 1 ns pulse width

# List of different time jitter values to test
time_jitter_values = [0, 0.1, 0.2, 0.5, 0.8, 1, 1.5, 2]  # Example time jitter values in ns
plt.figure(figsize=(10, 6))

# Run Monte Carlo simulations for each time jitter value
for time_jitter_ns in time_jitter_values:
    time_jitter_range = time_jitter_ns  # Standard deviation for normal distribution

    # Arrays to accumulate results
    tau_accum = None
    hom_accum = None

    for _ in range(n_simulations):
        time_jitter = np.random.normal(0, time_jitter_range, 1)
        tau, hom_result = sim(time_jitter, pulse_width_ns)
        if tau_accum is None:
            tau_accum = tau
            hom_accum = hom_result
        else:
            hom_accum += hom_result

    # Average the results
    hom_avg = hom_accum  / n_simulations

    # Plot the averaged HOM interference for this time jitter value
    plt.plot(tau_accum, hom_avg, label=f'Time Jitter = {time_jitter_ns} ns')
plt.legend()
plt.title('HOM Measurement using Gaussian Pulse with Various Time Jitters (Monte Carlo Simulation)')
plt.xlabel('Time [ns]')
plt.ylabel('Coincidence Amplitude')
plt.grid(True)
plt.show()
# %%
