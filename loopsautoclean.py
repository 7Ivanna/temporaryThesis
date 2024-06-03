import numpy as np
import math
from scipy.constants import physical_constants
import sys
import os
import os.path



def interference_loss(mesh_size, phases, photon_width, switch_time, time_jitter):
     # User Settings Here
    #N = 6
    N = mesh_size
    r= N*(N+2)
    skips = int(.5*N-1)
    ref_index = 1.4677 # based on paper jacob sent
    # photon_width = 1e-9 # 1 nanosec 
    #switch_time = 2e-11 #20 picoseconds fpr the rise time , etc
    time_jitter=time_jitter*photon_width
    total_timer = photon_width+switch_time+time_jitter
    speed_of_light = physical_constants['speed of light in vacuum'][0] 
    modes = list(range(1, N + 1))
    points = N/2
    sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
    print(sorted_modes)
    top_divide = N // 2
    bottom_divide = N // 2 + 1
    top_array = np.zeros(top_divide, dtype=object)
    bottom_array = np.zeros(bottom_divide, dtype=object)
    totaltimebins=0


    # calculate length of optic cable needed
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
    print(sorted_modes)



    print('Step \t Time-Bin \t Theta \t Phi')
    print("0 |   " , "bin", "|", "theta", "|", "phi")
    def table(step,bin,theta,phi):
    # print(f'{step:d} \t {bin:d} \t {theta:d} \t {phi:d} \t')

        print(step, "|\t" , bin, "|\t", theta, "|\t", phi)
        return 

    def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, count, top_divide,phases,points,skips,totaltimebins):
        top = np.zeros(top_divide, dtype=object)
        switchmode=0
        for i in range(N):
            table(totaltimebins,totaltimebins,0,0)
            totaltimebins=totaltimebins+1
            count = 0
            N = N
            array_length = len(bottom_array)
            middle_index = int((array_length - 2) )
            # move elements to the right in the bottom array
            bottom_array[1:] = bottom_array[:-1]  # shift elements to the right
            if len(sorted_modes) == N:  # check if sorted_modes length is equal to N
                bottom_array[0] = sorted_modes[i % N]  # add the new element at the beginning
            else:  # change sorted_modes to a length of 3 if it's not already
                sorted_modes = sorted_modes[:3]  # keep only the first 3 elements
            middle_value = bottom_array[middle_index]
            if middle_value != 0:
                middle_value,top = sort(N, middle_value, top_divide, count, top)
                bottom_array[middle_index] = middle_value
                sorted_modes[i % N] = middle_value
            count += 1   
        void_bin = bottom_array[-1]
        bottom_array[1:] = bottom_array[:-1]  # shift elements to the right
        bottom_array[0] = void_bin
        phases=phases
        Ncounter=0
        top, bottom_array, switchmode,totaltimebins,thetacount, phicount = interfere(bottom_array,top,phases,N, switchmode,totaltimebins)
        for i in range(int(N/2)-1) :
            top, bottom_array, switchmode,Ncounter,totaltimebins,thetacount,phicount = switch(top, bottom_array,N, switchmode,Ncounter,skips,totaltimebins,phases,thetacount, phicount)
        top, bottom_array, switchmode,totaltimebins=out( top, bottom_array, switchmode,totaltimebins)
        return
    # define the middle function
    def sort(N, value, top_divide, count, top):
        if value % 2 != 0:
            top[1:] = top[:-1]  # shift elements up by one position
            top[0] = value  # assign the new value to the first index
            return 0 ,top
        else:
            last_value = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value  # Assign the last value to the first index
            return value, top
        

    def interfere(bottom_array,top,phases,N,switchmode,totaltimebins):
        thetacount=0
        phicount=1
        if points<0: 
            points== 0
        for i in range (int(N-points)):
            theta=phases[thetacount]
            phi=phases[phicount]
            table(totaltimebins,totaltimebins,theta,phi)
            last_value_top = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = last_value_top  # Assign the last value to the first index
            last_value_bottom = bottom_array[-1]  # Store the last value of top
            bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
            bottom_array[0] = last_value_bottom  # Assign the last value to the first index
            totaltimebins=totaltimebins+1
            thetacount=thetacount+2
            phicount=phicount+2

        switchmode = switchmode + 1
        return top, bottom_array, switchmode,totaltimebins,thetacount, phicount


    def switch(top, bottom_array,N, switchmode,Ncounter,skips,totaltimebins,phases,thetacount,phicount): # could add a for looop for number of N that iterates thru the interference array
        counter=0
        totaltimebins=totaltimebins+1
        theta=phases[thetacount]
        phi=phases[phicount]
        if top[0] ==0 or bottom_array[-1] ==0:
            table(totaltimebins,totaltimebins,0,0)
        else:
            table(totaltimebins,totaltimebins,theta,phi)
            thetacount=thetacount+2
            phicount=phicount+2
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
        thethacount=thetacount-2
        phicount=phicount
        totaltimebins=totaltimebins+1
        for i in range(int(N-skips)): 
                theta=phases[thetacount]
                phi=phases[phicount]
                if top[0] ==0 or bottom_array[-1] ==0:
                    table(totaltimebins,totaltimebins,0,0)
                else:
                    table(totaltimebins,totaltimebins,theta,phi)
                    thetacount=thetacount+2
                    phicount=phicount+2
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
                totaltimebins=totaltimebins+1
        switchmode = switchmode + 1   
        Ncounter=Ncounter+1
        theta=phases[thetacount]
        phi=phases[phicount]
        table(totaltimebins,totaltimebins,theta,phi)
        thetacount=thetacount+2
        phicount=phicount+2
        top_temp=top[0]
        bottom_temp=bottom_array[-1]
        top_temp = top[0]  # Store the last value of top
        top[0] = bottom_temp  # Assign the last value to the first index
        bottom_array[-1] = top_temp  # Store the last value of top
        if Ncounter== N/2+1:
            return top, bottom_array, switchmode,Ncounter,totaltimebins
        else:
            for i in range(int(N/2)):
                last_value_top = top[-1]  # Store the last value of top
                top[1:] = top[:-1]  # Shift elements up by one position
                top[0] = last_value_top  # Assign the last value to the first index
                last_value_bottom = bottom_array[-1]  # Store the last value of top
                bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
                bottom_array[0] = last_value_bottom  # Assign the last value to the first index
                totaltimebins=totaltimebins+1
                if top[0] ==0 or bottom_array[-1] ==0:
                    table(totaltimebins,totaltimebins,0,0)
                else:
                    max=len(phases)
                    if phicount/max >=1:
                        theta=0
                        phi=0
                    else:
                        theta=phases[thetacount]
                        phi=phases[phicount]
                        thetacount=thetacount+2
                        phicount=phicount+2
                    table(totaltimebins,totaltimebins,theta,phi) 
                top_temp=top[-1]
                counter=counter+1 
                top_temp=top[0]
                bottom_temp=bottom_array[-1]
                top[0] = bottom_temp  # Assign the last value to the first index
                bottom_array[-1] = top_temp  # Store the last value of top   
            top_temp=top[0]
            bottom_temp=bottom_array[-1]
            top[0] = bottom_temp  # Assign the last value to the first index
            bottom_array[-1] = top_temp  # Store the last value of top
            return top, bottom_array, switchmode, Ncounter,totaltimebins,thetacount,phicount
        
    def out(top, bottom_array, switchmode,totaltimebins):
        print("outnow")
        totaltimebins=totaltimebins+1
        lossbins=totaltimebins
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
        saved_numbers = []  # List to save the numbers coming from bottom_array[-1]
        for _ in range(N+1):  # Loop for the length of bottom_array - 1
            saved_numbers.append(bottom_array[-2])  # Save the number coming from bottom_array[-1]
            theta=0
            phi=0
            table(totaltimebins,totaltimebins,theta,phi)
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
            totaltimebins=totaltimebins+1    
        print("Cable bins traversed:",lossbins)
        return top, bottom_array, switchmode,totaltimebins# Return the modified variables


    up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, 0, top_divide,phases,points,skips,totaltimebins)
    print(speed_of_light)
    print("speed in cable",speed_in_cable)
    print("top length",perfect_top)
    print("bottom length", perfect_bottom)
    print("total length", perfect_top+perfect_bottom)

    # loss time 

    alpha_mzi = 0.01145
    alpha_fibre = 0.0000460506
    total_fibre = perfect_top+perfect_bottom
    mzi_loss_even = (np.sqrt(1-alpha_mzi))**N # loss for even modes
    mzi_loss_odd = (np.sqrt(1-alpha_mzi))**(N+1) # loss for odd modes
    fibre_loss =  (np.sqrt(1-alpha_fibre))**(total_fibre)

    total_loss_odd = mzi_loss_odd*fibre_loss
    total_loss_even = mzi_loss_even*fibre_loss
    print(total_fibre)
    average_loss =(N/2*total_loss_odd+N/2*total_loss_even)/N
    print("loss",(1-average_loss)*100)
    return  total_fibre


# Calculate interference loss for each N value
loss_results = {}
#for N in N_values:
   # loss_results[N] = interference_loss(mesh_size, phases, photon_width, rise_time, time_jitter)

# Save results to a text file
with open("totalFibreNeeded1.txt", "w") as file:
    for N, loss in loss_results.items():
        file.write(f"N={N}: Loss={loss:.16f}\n")

# Define the mesh size, phases array, photon width, switch rise time, and the ammount of photon time jitter (decimal) here
mesh_size = 6
r= mesh_size*(mesh_size+2)
phases = np.arange(1, r+1)
photon_width = 1e-9 # 1 nanosec 
rise_time = 2e-11 #20 picoseconds fpr the rise time , etc
time_jitter=0.5
loss_results[mesh_size] = interference_loss(mesh_size, phases, photon_width, rise_time, time_jitter)