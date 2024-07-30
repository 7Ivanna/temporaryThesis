#please code gods let this work
import numpy as np
import math
from tabulate import tabulate
from scipy.constants import physical_constants
# User Settings Here
N = 4
r= N**2-N
phases = np.arange(1, r+1)
skips = int(.5*N-1)
ref_index = 1.4677 # based on paper jacob sent
photon_width = 1e-9 # 1 nanosec 
switch_time = 2e-11 #20 picoseconds fpr the rise time , etc
time_jitter=0.5*photon_width
total_timer=photon_width+switch_time+time_jitter

speed_of_light = physical_constants['speed of light in vacuum'][0] 

divider = 4/N
modes = list(range(1, N + 2))
points = N/2
sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)
Ncounter = 0
# Initialize current_mode with the first odd number
current_mode = 1
totaltimebins=0
# Set the step size to alternate between odd and even numbers
step_size = 1

# calculate length of optic cable needed
total_divisions = N + 1
speed_of_light = physical_constants['speed of light in vacuum'][0] 
speed_in_cable = speed_of_light/ref_index
perfect_top=(top_divide)*speed_in_cable
perfect_bottom=bottom_divide*speed_in_cable

# not perfect stuff now
perfect_top=(top_divide)*speed_in_cable*total_timer
perfect_bottom=bottom_divide*speed_in_cable*total_timer


timebins = N * bottom_divide
array_length = len(bottom_array)
print(array_length)
middle_index = (array_length - 1) // 2
midlength=N
print(sorted_modes)

print('Step \t Time-Bin \t Theta \t Phi')
print("0 |   " , "bin", "|", "theta", "|", "phi")

def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, count, top_divide,phases,points,skips,totaltimebins):
    top = np.zeros(top_divide, dtype=object)

    shifted_values = np.zeros(bottom_divide)  # Array to store shifted out values
    switchmode=0
    for i in range(N):
        table(totaltimebins,totaltimebins,0,0)
        totaltimebins=totaltimebins+1
        #print(f"ts={i + 1}")
        bottom_index = i % bottom_divide
        count = 0
        N = N
        array_length = len(bottom_array)
       # print(array_length)
        middle_index = int((array_length - 2) )
       # print("middle",middle_index)
        # Move elements to the right in the bottom array
        bottom_array[1:] = bottom_array[:-1]  # Shift elements to the right
        
        if len(sorted_modes) == N:  # Check if sorted_modes length is equal to N
            bottom_array[0] = sorted_modes[i % N]  # Add the new element at the beginning
        else:  # Change sorted_modes to a length of 3 if it's not already
            sorted_modes = sorted_modes[:3]  # Keep only the first 3 elements
            
        #print("bottom loop:", bottom_array)
        
        middle_value = bottom_array[middle_index]
        if middle_value != 0:
            middle_value,top = sort(N, middle_value, top_divide, count, top)
            bottom_array[middle_index] = middle_value
            sorted_modes[i % N] = middle_value

        count += 1   
        #print("-------------totaltimebins",totaltimebins)
        
   # print("topper?",top, "i just met er")
    void_bin = bottom_array[-1]
    bottom_array[1:] = bottom_array[:-1]  # Shift elements to the right
    bottom_array[0] = void_bin
   # print("bottomloop", bottom_array)
   # print("-------------")
    phases=phases
    Ncounter=0
    top, bottom_array, switchmode,totaltimebins,thetacount, phicount = interfere(bottom_array,top,phases,N, switchmode,totaltimebins)
    for i in range(int(N/2)-1) :
        top, bottom_array, switchmode,Ncounter,totaltimebins,thetacount,phicount = switch(top, bottom_array,N, switchmode,Ncounter,skips,totaltimebins,phases,thetacount, phicount)
    top, bottom_array, switchmode,totaltimebins=out( top, bottom_array, switchmode,totaltimebins)
    return
# Define the middle function
def sort(N, value, top_divide, count, top):
    if value % 2 != 0:
       # print(value)
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = value  # Assign the new value to the first index
       # print(f"Middle value: {value}")
       # print("Top loop:",top)
        
        return 0 ,top
    else:
        last_value = top[-1]  # Store the last value of top
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = last_value  # Assign the last value to the first index
       # print("Top array:",top)
  
        return value, top