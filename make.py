#please code gods let this work
import numpy as np
import math
from tabulate import tabulate
from scipy.constants import physical_constants
# User Settings Here
N = 6
skips = 1 # need to changethis based on N 1 if 4 2 if 6
ref_index = 1.4677 # based on paper jacob sent
photon_width = 1e-9 # 1 nanosec 


speed_of_light = physical_constants['speed of light in vacuum'][0] 

divider = 4/N
modes = list(range(1, N + 1))
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
perfect_top=(top_divide)*speed_in_cable*photon_width
perfect_bottom=bottom_divide*speed_in_cable*photon_width


timebins = N * bottom_divide
array_length = len(bottom_array)
print(array_length)
middle_index = (array_length - 1) // 2
midlength=N
print(sorted_modes)
phases=[1,2]

def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, count, top_divide,phases,points,skips,totaltimebins):
    top = np.zeros(top_divide, dtype=object)
    shifted_values = np.zeros(bottom_divide)  # Array to store shifted out values
    switchmode=0
    for i in range(N):
        totaltimebins=totaltimebins+1
        print(f"ts={i + 1}")
        bottom_index = i % bottom_divide
        count = 0
        N = N
        array_length = len(bottom_array)
        print(array_length)
        middle_index = int((array_length - 2) )
        print("middle",middle_index)
        # Move elements to the right in the bottom array
        bottom_array[1:] = bottom_array[:-1]  # Shift elements to the right
        
        if len(sorted_modes) == N:  # Check if sorted_modes length is equal to N
            bottom_array[0] = sorted_modes[i % N]  # Add the new element at the beginning
        else:  # Change sorted_modes to a length of 3 if it's not already
            sorted_modes = sorted_modes[:3]  # Keep only the first 3 elements
            
        print("bottom loop:", bottom_array)
        
        middle_value = bottom_array[middle_index]
        if middle_value != 0:
            middle_value,top = sort(N, middle_value, top_divide, count, top)
            bottom_array[middle_index] = middle_value
            sorted_modes[i % N] = middle_value

        count += 1   
        print("-------------totaltimebins",totaltimebins)

    print("topper?",top, "i just met er")
    void_bin = bottom_array[-1]
    bottom_array[1:] = bottom_array[:-1]  # Shift elements to the right
    bottom_array[0] = void_bin
    print("bottomloop", bottom_array)
    print("-------------")
    phases=phases
    Ncounter=0
    top, bottom_array, switchmode,totaltimebins = interfere(bottom_array,top,phases,N, switchmode,totaltimebins)
    for i in range(int(N/2)-1) :
        top, bottom_array, switchmode,Ncounter,totaltimebins = switch(top, bottom_array,N, switchmode,Ncounter,skips,totaltimebins)
    top, bottom_array, switchmode,totaltimebins=out( top, bottom_array, switchmode,totaltimebins)
    return
# Define the middle function
def sort(N, value, top_divide, count, top):
    if value % 2 != 0:
        print(value)
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = value  # Assign the new value to the first index
        print(f"Middle value: {value}")
        print("Top loop:",top)
        
        return 0 ,top
    else:
        last_value = top[-1]  # Store the last value of top
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = last_value  # Assign the last value to the first index
        print("Top array:",top)

        return value, top
    

def interfere(bottom_array,top,phases,N,switchmode,totaltimebins):
    print("---inerfere---")
    print("TotalBins",totaltimebins)
    if points<0: 
        points== 0
    for i in range (int(N-points)):
        print("Top loop:",top)
        print("bottom loop:", bottom_array)
        print("modes interfering:", top[0], "and", bottom_array[-1])
        print("TotalBins",totaltimebins)
        print("---continue---")
        last_value_top = top[-1]  # Store the last value of top
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = last_value_top  # Assign the last value to the first index
        last_value_bottom = bottom_array[-1]  # Store the last value of top
        bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
        bottom_array[0] = last_value_bottom  # Assign the last value to the first index
        print("Top loop:",top)
        print("bottom loop:", bottom_array)
        print("---new loop---") 
        totaltimebins=totaltimebins+1
    switchmode = switchmode + 1
    print(switchmode)
    return top, bottom_array, switchmode,totaltimebins


def switch(top, bottom_array,N, switchmode,Ncounter,skips,totaltimebins): # could add a for looop for number of N that iterates thru the interference array
    counter=0
    
    print("---switch---", switchmode)
    print("TotalBins",totaltimebins)
    print(top)
    print(bottom_array)
    print("modes interfering:", top[0], "and", bottom_array[-1])
    top_temp=top[0]
    bottom_temp=bottom_array[-1]
    top[0]=bottom_temp
    bottom_array[-1] = top_temp  # Store the last value of top
    last_value_top = top[-1]  # Store the last value of top
    top[1:] = top[:-1]  # Shift elements up by one position
    top[0] = last_value_top  # Assign the last value to the first index
    last_value_bottom = bottom_array[-1]  # Store the last value of top
    bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
    bottom_array[0] = last_value_bottom  # Assign the last value to the first index
    print("Top array:",top)
    print("bot array:",bottom_array)
   # if switchmode % 2 != 0 :
    for i in range(int(N-skips)): #need to changethis based on N 1 if 4 2 if 6-------------------------------------------------------------------------------
            print("forloop")
            totaltimebins=totaltimebins+1
            print("modes interfering:", top[0], "and", bottom_array[-1])
            print("TotalBins",totaltimebins)
            top_temp=top[0]
            bottom_temp=bottom_array[-1]
            top[0]=bottom_temp
            bottom_array[-1] = top_temp  # Store the last value of top
            last_value_top = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value_top  # Assign the last value to the first index
            last_value_bottom = bottom_array[-1]  # Store the last value of top
            bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
            bottom_array[0] = last_value_bottom  # Assign the last value to the first index
            print("Top array:",top)
            print("bot array:",bottom_array)
            
    switchmode = switchmode + 1   
    Ncounter=Ncounter+1
    print(top)
    print(bottom_array)
    totaltimebins=totaltimebins+1
    print("modes interfering:", top[0], "and", bottom_array[-1])
    print("TotalBins",totaltimebins)
    top_temp=top[0]
    bottom_temp=bottom_array[-1]
    print("toptemp",top_temp)
    print("bottom_temp",bottom_temp)
    top_temp = top[0]  # Store the last value of top
    top[0] = bottom_temp  # Assign the last value to the first index
    print("toptemp",top_temp)

    bottom_array[-1] = top_temp  # Store the last value of top
    print("Top array:",top)
    print("bot array:",bottom_array)
    print("---------------------secondloop")
    print(Ncounter)
    if Ncounter== N/2+1:
        return top, bottom_array, switchmode,Ncounter,totaltimebins
    else:
        for i in range(int(N/2)):
            print("TotalBins",totaltimebins)
            last_value_top = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value_top  # Assign the last value to the first index
            last_value_bottom = bottom_array[-1]  # Store the last value of top
            bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
            bottom_array[0] = last_value_bottom  # Assign the last value to the first index
            print("Top array:",top)
            print("bot array:",bottom_array)
            totaltimebins=totaltimebins+1
            print("modes interfering:", top[0], "and", bottom_array[-1])
            print("TotalBins",totaltimebins)   
            top_temp=top[-1]
            counter=counter+1
            print("count",counter)    
            top_temp=top[0]
            bottom_temp=bottom_array[-1]
            top[0] = bottom_temp  # Assign the last value to the first index
            bottom_array[-1] = top_temp  # Store the last value of top
            #IM GOING TO SCREAM  
            
        top_temp=top[0]
        bottom_temp=bottom_array[-1]
        top[0] = bottom_temp  # Assign the last value to the first index
        bottom_array[-1] = top_temp  # Store the last value of top
        
        return top, bottom_array, switchmode, Ncounter,totaltimebins
def out(top, bottom_array, switchmode,totaltimebins):
    print("outnow")
    print("Top array:", top)
    print("bot array:", bottom_array)
    print("TotalBins",totaltimebins)
    top_temp = top[0]
    bottom_temp = bottom_array[-1]
    top[0] = bottom_temp  # Assign the last value to the first index
    bottom_array[-1] = top_temp  # Store the last value of top
    last_value_top = top[-1]  # Store the last value of top
    top[1:] = top[:-1]  # Shift elements up by one position
    top[0] = last_value_top  # Assign the last value to the first index
    last_value_bottom = bottom_array[-1]  # Store the last value of top
    bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
    bottom_array[0] = last_value_bottom  # Assign the last value to the first index
    print("Top array:", top)
    print("bot array:", bottom_array)
    print("modes interfering:", top[0], "and", bottom_array[-1])
    saved_numbers = []  # List to save the numbers coming from bottom_array[-1]
    for _ in range(N+1):  # Loop for the length of bottom_array - 1
        saved_numbers.append(bottom_array[-2])  # Save the number coming from bottom_array[-1]
        print("modes interfering:", top[0], "and", bottom_array[-1])
        bottom_array[-2]=0
        top_temp = top[0]
        bottom_temp = bottom_array[-1]
        top[0] = bottom_temp  # Assign the last value to the first index
        bottom_array[-1] = top_temp  # Store the last value of top
        last_value_top = top[-1]  # Store the last value of top
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = last_value_top  # Assign the last value to the first index
        last_value_bottom = bottom_array[-1]  # Store the last value of top
        bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
        bottom_array[0] = last_value_bottom  # Assign the last value to the first i
        #top[-1] = 0
          # Move the last element of bottom_array to the beginning of top
        print("Top array:", top)
        print("bot array:", bottom_array)
        print("saved_numbers:", saved_numbers)
        totaltimebins=totaltimebins+1
        print("TotalBins",totaltimebins)
        



    return top, bottom_array, switchmode,totaltimebins # Return the modified variables


up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, 0, top_divide,phases,points,skips,totaltimebins)
print(speed_of_light)
print("speed in cable",speed_in_cable)
print("top length",perfect_top)
print("bottom length", perfect_bottom)
print("total length", perfect_top+perfect_bottom)

freq=speed_in_cable/(1550e-9)
print(1/freq)