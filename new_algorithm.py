import numpy as np
import math
from tabulate import tabulate
from scipy.constants import physical_constants
# User Settings Here
N = 4  # replace with number of modes 
r= N*(N+2)
phases = np.arange(1, r+1) # fill array with desired phases in standard order of operation
ref_index = 1.4677 # Generally accepted value
pulse_width = 1e-9 # 1 nanosec 
switch_time = 2e-11 #20 picoseconds fpr the rise time , etc
time_jitter=0.5*pulse_width
total_timer=pulse_width+switch_time+time_jitter

speed_of_light = physical_constants['speed of light in vacuum'][0] 
skips = int(.5*N-1)
divider = 4/N
modes = list(range(1, N + 1))
points = N/2
sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)
Ncounter = 0
totaltimebins=0

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
top = np.zeros(top_divide, dtype=object)

print(sorted_modes)

print('Step \t Time-Bin \t Theta \t Phi')
print("0 |   " , "bin", "|", "theta", "|", "phi")

def table(step,bin,theta,phi):
    print(step, "|\t" , bin, "|\t", theta, "|\t", phi)
    return 

def entry(N, bottom_divide, sorted_modes, totaltimebins, top,phases,skips):
    entry_steps = int(N / 2 +1)
    bottom = np.zeros(bottom_divide, dtype=object)
    count = 0 
    thetacount=0
    phicount=1
    bottom[0] = sorted_modes[0]  # Move the first element to bottom[0]
    table(totaltimebins, totaltimebins, 0, 0)
    for i in range(entry_steps-1):
        totaltimebins += 1
        bottom[0] = sorted_modes[0]  # Move the first element to bottom[0]
        bottom, top = entrance_x(bottom, top, count)
        sorted_modes = sorted_modes[1:] + [0]  # Move each element up by one position and add zero at the end
        table(totaltimebins, totaltimebins, 0, 0)
        count += 1 
    print(bottom)
    sorted_modes = sorted_modes[:-skips]
    bottom, top, count,phases,totaltimebins,entry_steps, thetacount, phicount =interfere_x(sorted_modes, top, count,phases,totaltimebins,entry_steps)
    bottom, top, count,phases,totaltimebins,entry_steps, thetacount, phicount=switch_void(sorted_modes, top, count,phases,totaltimebins, int(N/2), thetacount, phicount)
    bottom, top, count,phases,totaltimebins,entry_steps, thetacount, phicount=switch_x(sorted_modes, top, count,phases,totaltimebins,int(N/2 +1), thetacount, phicount)
   # bottom, top, count,phases,totaltimebins,entry_steps=switch_void(sorted_modes, top, count,phases,totaltimebins, int(N/2))
   # bottom, top, count,phases,totaltimebins,entry_steps=switch_x(sorted_modes, top, count,phases,totaltimebins,int(N/2 +1))
   

    #bottom, top, count,phases,totaltimebins,entry_steps=switch_void(sorted_modes, top, count,phases,totaltimebins, skips)
    exit(bottom, top, count,phases,totaltimebins,entry_steps,N)

    

    return 

def entrance_x(input, top,count):
    # Move the elements of the top array one position to the right
    top[1:] = top[:-1]
    top[0] = 0
    # Place the first element of input into the first position of top
   
    print(f"input[0]: {input[0]}")
    print(f"top array: {top}")
    print(f"modes interfering: top[0] and input[0]: {top[0]} and {input[0]}")
    top[0] = input[0]
    
    print(f"top array: {top}")
    
    return input, top

def interfere_x(bottom, top, count,phases,totaltimebins,entry_steps):
        thetacount=0
        phicount=1
        for i in range(entry_steps):
            totaltimebins=totaltimebins+1
            last_value_top = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value_top  # Assign the last value to the first index
            print(top)
            print(bottom)
            print(f"modes interfering: top[0] and input[0]: {top[0]} and {bottom[0]}")
            print("no this one")
            if top[0] and bottom[0] 
            theta=phases[thetacount]
            phi=phases[phicount]
            
            table(totaltimebins,totaltimebins,theta,phi)


            table(totaltimebins, totaltimebins, 0, 0)
            last_value_bottom = bottom[0]  # Store the first value of bottom
            bottom[:-1] = bottom[1:]  # Shift elements down by one position
            bottom[-1] = last_value_bottom  # Assign the first value to the last index
            thetacount=thetacount+2
            phicount=phicount+2
 
        last_value_bottom = bottom[-1]  # Store the first value of bottom
        bottom[1:] = bottom[:-1]  # Shift elements down by one position
        bottom[0] = last_value_bottom  # Assign the first value to the last index
        bottom_temp=bottom[0]
        print(bottom)
        top_temp = top[0]  # Store the last value of top
        print(top_temp)
        print(bottom[0])
        print("bottom_temp",bottom_temp)
        top_temp = top[0]  # Store the last value of top
        top[0] = bottom_temp  # Assign the last value to the first index
        bottom[0]=top_temp
        print(top)
        print(bottom)
        print(bottom , top, "hehe")
        
        return bottom, top, count,phases,totaltimebins,entry_steps, thetacount, phicount

def switch_void(bottom, top, count,phases,totaltimebins,entry_steps, thetacount, phicount):
    for i in range(entry_steps):
        last_value_bottom = bottom[0]  # Store the first value of bottom
        bottom[:-1] = bottom[1:]  # Shift elements down by one position
        bottom[-1] = last_value_bottom  # Assign the first value to the last in
        last_value_top = top[-1]  # Store the last value of top
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = last_value_top  # Assign the last value to the first index
        print(top)
        print(bottom)
        totaltimebins=totaltimebins+1
        print(f"modes interfering: top[0] and input[0]: {top[0]} and {bottom[0]}")
        table(totaltimebins, totaltimebins, 0, 0)
        print(top)
        print(bottom)
        bottom_temp=bottom[0]
        print(bottom)
        top_temp = top[0]  # Store the last value of top
        print(top_temp)
        print(bottom[0])
        print("bottom_temp",bottom_temp)
        top_temp = top[0]  # Store the last value of top
        top[0] = bottom_temp  # Assign the last value to the first index
        bottom[0]=top_temp
        print(top)
        print(bottom)
        print(top, bottom)
    last_value_bottom = bottom[0]  # Store the first value of bottom
    bottom[:-1] = bottom[1:]  # Shift elements down by one position
    bottom[-1] = last_value_bottom  # Assign the first value to the last in
    print(bottom)
    '''top_temp = top[0]  # Store the last value of top
    print(top_temp)
    print(bottom[0])
    print("bottom_temp",bottom_temp)
    top_temp = top[0]  # Store the last value of top
    top[0] = bottom_temp  # Assign the last value to the first index
    bottom[0]=top_temp
    print(top)
    print(bottom)
    print(bottom , top, "hehe")'''

    return bottom, top, count,phases,totaltimebins,entry_steps,thetacount, phicount

def switch_x(bottom, top, count,phases,totaltimebins,entry_steps, thetacount, phicount):
    for i in range(entry_steps):
        print('start')
        totaltimebins=totaltimebins+1
        last_value_top = top[-1]  # Store the last value of top
        top[1:] = top[:-1]  # Shift elements up by one position
        top[0] = last_value_top  # Assign the last value to the first index
        print(top)
        print(bottom)
        print(f"modes interfering: top[0] and input[0]: {top[0]} and {bottom[0]}")
        table(totaltimebins, totaltimebins, 0, 0)
        theta=phases[thetacount]
        phi=phases[phicount]
        table(totaltimebins, totaltimebins, theta, phi)
        print("its this one")
        bottom_temp=bottom[0]
        print(bottom)
        print(top)
        top_temp = top[0]  # Store the last value of top
        print(top_temp)
        print(bottom[0])
        print("bottom_temp",bottom_temp)
        top[0] = bottom_temp  # Assign the last value to the first index
        bottom[0]=top_temp
        
        last_value_bottom = bottom[0]  # Store the first value of bottom
        bottom[:-1] = bottom[1:]  # Shift elements down by one position
        bottom[-1] = last_value_bottom  # Assign the first value to the last in
        print(top)
        print(bottom)
       
        thetacount=thetacount+2
        phicount=phicount+2
    last_value_bottom = bottom[-1]  # Store the first value of bottom
    bottom[1:] = bottom[:-1]  # Shift elements down by one position
    bottom[0] = last_value_bottom  # Assign the first value to the last index
    
    
    return bottom, top, count,phases,totaltimebins,entry_steps,thetacount, phicount

def exit_interference(bottom, top, count,phases,totaltimebins,entry_steps,exit_sum,N,out,i):
        if  exit_sum ==0 or exit_sum ==1:
            print(bottom)
            print('start')
            totaltimebins=totaltimebins+1
            last_value_top = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value_top  # Assign the last value to the first index
            last_value_bottom = bottom[0]  # Store the first value of bottom
            bottom[:-1] = bottom[1:]  # Shift elements down by one position
            bottom[-1] = last_value_bottom  # Assign the first value to the last 
            print(top)
            print(bottom)
            print(f"modes interfering: top[0] and input[0]: {top[0]} and {bottom[0]}")
            table(totaltimebins, totaltimebins, 0, 0)

            bottom_temp=bottom[0]
            print(bottom)
            print(top)
            top_temp = top[0]  # Store the last value of top
            print(top_temp)
            print(bottom[0])
            print("bottom_temp",bottom_temp)

            print(top)
            print(bottom)
            print(bottom)
            s=int(i+entry_steps)
            print(s)
    
            out[i]=bottom[-1]
            bottom[-1] = 0
            print(bottom)
            print(top)
            print(out)
            entry_steps-=entry_steps
            return bottom, top, count,phases,totaltimebins,entry_steps,exit_sum,N,out,i
        
        else:
            print(bottom)
            print('start else')
            totaltimebins=totaltimebins+1
            last_value_top = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value_top  # Assign the last value to the first index
            last_value_bottom = bottom[0]  # Store the first value of bottom
            bottom[:-1] = bottom[1:]  # Shift elements down by one position
            bottom[-1] = last_value_bottom  # Assign the first value to the last 
            print(top)
            print(bottom)
            print(f"modes interfering: top[0] and input[0]: {top[0]} and {bottom[0]}")
            table(totaltimebins, totaltimebins, 0, 0)

            bottom_temp=bottom[0]
            print(bottom)
            print(top)
            top_temp = top[0]  # Store the last value of top
            print(top_temp)
            print(bottom[0])
            print("bottom_temp",bottom_temp)
            top[0] = bottom_temp  # Assign the last value to the first index
            bottom[0]=top_temp
            print(top)
            print(bottom)
            print(bottom)
            s=int(i+entry_steps)
            print(s)
    
            out[i]=bottom[-1]
            bottom[-1] = 0
            print(bottom)
            print(top)
            print(out)
            entry_steps-=entry_steps
            return bottom, top, count,phases,totaltimebins,entry_steps,exit_sum,N,out,i

def exit(bottom, top, count,phases,totaltimebins,entry_steps,N):
    print("exit function")
    print( top)
    print(bottom)
    out=np.zeros(N+2, dtype=object)
    print(out)
    exit_sum = 0
    for i in range(int(np.size(bottom)+N/2)):
        if exit_sum==2:
            print("haaaaaa")
            '''
            totaltimebins=totaltimebins+1
            last_value_top = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value_top  # Assign the last value to the first index
            last_value_bottom = bottom[0]  # Store the first value of bottom
            bottom[:-1] = bottom[1:]  # Shift elements down by one position
            bottom[-1] = last_value_bottom  # Assign the first value to the last 
            print(top)
            print(bottom)
            print(f"modes interfering: top[0] and input[0]: {top[0]} and {bottom[0]}")
            table(totaltimebins, totaltimebins, 0, 0)
            bottom_temp=bottom[0]
            print(bottom)
            print(top)
            top_temp = top[0]  # Store the last value of top
            print(top_temp)
            print(bottom[0])
            print("bottom_temp",bottom_temp)
            top[0] = bottom_temp  # Assign the last value to the first index
            bottom[0]=top_temp
            out[i]=0
            exit_sum+=1
            print(bottom)
            print(top)
            print(out)
            entry_steps-=entry_steps'''
          
        print("exit sum",exit_sum)
        bottom, top, count,phases,totaltimebins,entry_steps,exit_sum,N,out,i=exit_interference(bottom, top, count,phases,totaltimebins,entry_steps,exit_sum,N,out,i)
        print(bottom)
        
        exit_sum+=1
        print("exit sum",exit_sum)

             

    return

entry(N,bottom_divide,sorted_modes,totaltimebins,top,phases,skips)
