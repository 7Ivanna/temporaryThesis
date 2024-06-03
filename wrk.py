#please code gods let this work
import numpy as np
import math

N = 4
divider = 4/N
modes = list(range(1, N + 1))
points= 2
sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 1

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2
midlength=N
print(sorted_modes)
phases=[1,2]

def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, count, top_divide,phases,points):
    top = np.zeros(top_divide, dtype=object)
    shifted_values = np.zeros(bottom_divide)  # Array to store shifted out values
    switchmode=0
    for i in range(N):
        print(f"ts={i + 1}")
        bottom_index = i % bottom_divide
        count = 0
        N = N
        array_length = len(bottom_array)
        middle_index = ((array_length - 1) // 2)
        print(middle_index)
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
        print("-------------")

    print("topper?",top, "i just met er")
    void_bin = bottom_array[-1]
    bottom_array[1:] = bottom_array[:-1]  # Shift elements to the right
    bottom_array[0] = void_bin
    print("bottomloop", bottom_array)
    print("-------------")
    phases=phases
    top, bottom_array, switchmode = interfere(bottom_array,top,phases,N, switchmode)
    for i in range(int(N/2)) :
        top, bottom_array, switchmode = switch(top, bottom_array,N, switchmode)

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
    

def interfere(bottom_array,top,phases,N,switchmode):
    print("---inerfere---")
    if points<0: 
        points== 0
    for i in range (int(N-points)):
        print("Top loop:",top)
        print("bottom loop:", bottom_array)
        print("modes interfering:", top[0], "and", bottom_array[-1])
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
    switchmode = switchmode + 1
    print(switchmode)
    return top, bottom_array, switchmode


def switch(top, bottom_array,N, switchmode): # could add a for looop for number of N that iterates thru the interference array
    print("---switch---", switchmode)
    print(top)
    print(bottom_array)
    
   # if switchmode % 2 != 0 :
    for i in range(int(N/2+1)):
            print("modes interfering:", top[0], "and", bottom_array[-1])
            top_temp=top[-1]
            bottom_temp=bottom_array[0]
            print("toptemp",top_temp)
            print("bottom_temp",bottom_temp)
            top_temp = top[-1]  # Store the last value of top
            top[1:] = top[:-1]  # Shift elements up by one position
            top[0] = bottom_temp  # Assign the last value to the first index
            last_value_bottom = bottom_array[-1]  # Store the last value of top
            bottom_array[0] = top_temp  # Store the last value of top
            bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
            bottom_array[0] = last_value_bottom  # Assign the last value to the first index
            print("Top array:",top)
            print("bot array:",bottom_array)
    switchmode = switchmode + 1   
          #IM GOING TO SCREAM  
    return top, bottom_array, switchmode
  #else:
     #   for i in range(int(N/2+1)):
       #     print("modes interfering:", top[0], "and", bottom_array[-1])



      #      top_temp=top[-1]
      #      bottom_temp=bottom_array[0]
       #     print("toptemp",top_temp)
      #      print("bottom_temp",bottom_temp)
       #     top_temp = top[-1]  # Store the last value of top
        #    top[1:] = top[:-1]  # Shift elements up by one position
         #   top[0] = bottom_temp  # Assign the last value to the first index
          #  last_value_bottom = bottom_array[-1]  # Store the last value of top
           # bottom_array[0] = top_temp  # Store the last value of top
           # bottom_array[1:] = bottom_array[:-1]  # Shift elements up by one position
           # bottom_array[0] = last_value_bottom  # Assign the last value to the first index
           # print("toptemp",top_temp)

         #   print("Top array:",top)
       #     print("bot array:",bottom_array)'''
     #   switchmode = switchmode + 1
      #  return top, bottom_array, switchmode
    #
# Call the function
up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, 0, top_divide,phases,points)
