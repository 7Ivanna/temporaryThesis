import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def circlesplot(ax, photons, time_steps):
    N = 4  # Number of MZI modes (even only)
    top_divide = int(N / 2 + 1)
    bottom_divide = int(N / 2 + 2)
    bottom_points = np.linspace(0, 2*np.pi, bottom_divide)
    top_points = np.linspace(0, 2*np.pi, top_divide)
    bottom_r = 1
    top_r = 0.5
    rotation_bottom = np.pi/2 - bottom_points[1]
    rotation_top = np.pi/2 - top_points[1]
    bottom_x_points = bottom_r * np.cos(bottom_points + rotation_bottom)
    bottom_y_points = bottom_r * np.sin(bottom_points + rotation_bottom)
    top_x_points = top_r * np.cos(top_points - rotation_top)
    top_y_points = top_r * np.sin(top_points  - rotation_top)
    angles = np.linspace(0, 2*np.pi, 1000)
    bottom_x_circle = bottom_r * np.cos(angles)
    bottom_y_circle = bottom_r * np.sin(angles)
    top_x_circle = top_r * np.cos(angles)
    top_y_circle = top_r * np.sin(angles)

    ax.plot(bottom_x_circle, bottom_y_circle, color='purple')
    ax.plot(bottom_x_points, bottom_y_points, 'D', color='black', label='timebins')
    ax.plot(top_x_circle, top_y_circle+1.5, color='orange')
    ax.plot(top_x_points, top_y_points+1.5, 'D', color='black')
    
    # Plotting the paths of photons
    def animate(frame):
        ax.clear()
        ax.plot(bottom_x_circle, bottom_y_circle, color='purple')
        ax.plot(bottom_x_points, bottom_y_points, 'D', color='black', label='timebins')
        ax.plot(top_x_circle, top_y_circle+1.5, color='orange')
        ax.plot(top_x_points, top_y_points+1.5, 'D', color='black')
        for i, photon in enumerate(photons):
            if frame >= photon:
                if photon % 2 == 0:
                    ax.plot(bottom_x_points[photon//2 + 1], bottom_y_points[photon//2 + 1], 'o', color='red', label=f'Photon {i+1}')
                else:
                    ax.plot(top_x_points[photon//2], top_y_points[photon//2]+1.5, 'o', color='red', label=f'Photon {i+1}')
        ax.axis('equal')
        ax.axis('off')
        
        # Check if the diamonds of top and bottom loops are at (0, 1)
        if np.isclose(top_y_points[0] + 1.5, 1.0, atol=0.1):
            print(f"Top loop diamond is at (0, 1) at time step {frame}")
        if np.isclose(bottom_y_points[1], 1.0, atol=0.1):
            print(f"Bottom loop diamond is at (0, 1) at time step {frame}")

    anim = FuncAnimation(fig, animate, frames=time_steps, interval=1000, repeat=False)
    plt.show()
# Example usage
fig, ax = plt.subplots()
time_steps = 10  # Number of time steps
photons = [0, 1, 2, 3, 4]  # Array of photons
#circlesplot(ax, photons, time_steps)

import numpy as np

def move_photons(N):
    top_divide = N // 2 + 1
    bottom_divide = N // 2 + 2
    bottom_points = np.linspace(0, 2 * np.pi, bottom_divide)
    top_points = np.linspace(0, 2 * np.pi, top_divide)
    
    positions = []
    current_photon_index = 0
    photon_positions = [(f'bottom {1}', bottom_points[1])]  # Start with photon 1 at bottom[1]
    positions.append(list(photon_positions))  # Store initial positions
    
    for _ in range(1, N):
        # Move the current photon up by 1
        current_photon_index += 1
        photon_positions = [(f'bottom {i}', angle + 1) for i, (_, angle) in enumerate(photon_positions, 1)]
        # Add a new photon at the bottom
        photon_positions.insert(0, (f'bottom {current_photon_index + 1}', bottom_points[1]))
        positions.append(list(photon_positions))  # Store positions after moving and adding new photon
        
    return positions

# Example usage
N = 4  # Number of MZI modes
positions = move_photons(N)
print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")


import numpy as np

def move_photons(N):
    top_divide = N // 2 + 1
    bottom_divide = N // 2 + 2
    bottom_points = np.linspace(0, 2 * np.pi, bottom_divide)
    top_points = np.linspace(0, 2 * np.pi, top_divide)
    
    positions = []
    current_photon_index = 0
    photon_positions = [(f'bottom {1}', bottom_points[1])]  # Start with photon 1 at bottom[1]
    positions.append(list(photon_positions))  # Store initial positions
    
    for _ in range(1, N):
        # Move the current photon up by 1
        current_photon_index += 1
        if current_photon_index % 2 == 0:
            # Even-indexed photon stays in the bottom loop
            photon_positions = [(f'bottom {i}', angle + 1) for i, (_, angle) in enumerate(photon_positions, 1)]
            # Add a new photon at the bottom
            photon_positions.insert(0, (f'bottom {current_photon_index + 1}', bottom_points[1]))
        else:
            # Odd-indexed photon moves to the top loop
            photon_positions = [(f'top {i}', top_points[1]) for i, (_, angle) in enumerate(photon_positions, 1)]
            # Add a new photon at the bottom
            photon_positions.insert(0, (f'bottom {current_photon_index + 1}', bottom_points[1]))
        positions.append(list(photon_positions))  # Store positions after moving and adding new photon
        
    return positions

# Example usage

print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")
print("uuuuuuu")
#%%
import numpy as np
print("right")

N = 4  # Number of MZI modes
Np=N+1
bottom_divide = int(N / 2 + 2) 
bottom_loop = np.arange(N // 2 +1)
positions = move_photons(Np)
def move_photons(Np):
    N = 4 #number of mzi modes (even only)
    #divide circles into n+1 parts
    top_divide = int(N / 2 + 1) 
    bottom_divide = int(N / 2 + 2) 
    #create n+1 equally spaced angles
    bottom_points = np.linspace(0, 2*np.pi, bottom_divide)  
    top_points = np.linspace(0, 2*np.pi, top_divide)
    #set radii
    bottom_r = 1
    top_r = 0.5
    #rotation algorithm for the roatation angle so the bottom and top circles align at 0,1 
    rotation_bottom = np.pi/2 - bottom_points[1]
    
    rotation_top = np.pi/2 - top_points[1]
    #rotate time-bins for bottom points
    bottom_x_points = bottom_r * np.cos(bottom_points + rotation_bottom)
    bottom_y_points = bottom_r * np.sin(bottom_points + rotation_bottom)
    
    print(bottom_x_points,bottom_y_points)
    #rotate time-bins for top points
    top_x_points = top_r * np.cos(top_points - rotation_top)  
    top_y_points = top_r * np.sin(top_points  - rotation_top)
    print(top_x_points,top_y_points)
    positions = []
    photon_positions = [] 
    
    # input photon w corresponding label 
    for i in range(1, N + 1):
        if i % 2 == 0:
            photon_positions.append((f'bottom', bottom_points[1]))  #even
        else:
            photon_positions.append((f'top', bottom_points[1]))   #odd       
    positions.append(list(photon_positions))  # Store positions after introducing all photons

    if np.allclose(bottom_x_points, top_x_points) and np.allclose(bottom_y_points, top_y_points):
            print("Crossover detected at step", _)



    '''
    # Move all photons up by 1 at each step
    for _ in range(1, N):
        photon_positions = [(label, angle + 1) for label, angle in photon_positions]
        # Adjust positions for each photon to follow the desired pattern
        if _ < N - 1:
            photon_positions[_ + 1] = (photon_positions[_][0], bottom_points[1])
        positions.append(list(photon_positions))  # Store positions after moving
        '''
    return positions
long=int(len(bottom_loop))
print(long)
positions = []
print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")

#%%
import numpy as np
def move_photons(N):
    # Define the parameters for the circles
    bottom_r = 1
    top_r = 0.5
    bottom_points = np.linspace(0, 2 * np.pi, N // 2 + 2)
    top_points = np.linspace(0, 2 * np.pi, N // 2 + 1)
    
    # Calculate the rotation angle to align bottom and top circles at (0, 1)
    rotation_bottom = np.pi/2 - bottom_points[1]
    rotation_top = np.pi/2 - top_points[1]
    
    # rotate 
    bottom_x_points = bottom_r * np.cos(bottom_points + rotation_bottom)
    bottom_y_points = bottom_r * np.sin(bottom_points + rotation_bottom)
    top_x_points = top_r * np.cos(top_points - rotation_top)
    top_y_points = top_r * np.sin(top_points  - rotation_top)
    
    # Initialize photon positions
    positions = []
    photon_positions = []
    
    for i in range(1, N + 1):
        if i % 2 == 0:
            photon_positions.append((f'bottom', bottom_points[1]))  # Introduced as even
        else:
            photon_positions.append((f'top', bottom_points[1]))      # Introduced as odd
    
    positions.append(list(photon_positions))  # Store positions after introducing all photons
    
    # Move all photons up by 1 at each step
    for _ in range(1, N):
        # adjust positions for each photon to follow the desired pattern
        photon_positions = [(label, angle + 1) for label, angle in photon_positions]
        if _ < N - 1:
            photon_positions[_ + 1] = (photon_positions[_][0], bottom_points[1])
        positions.append(list(photon_positions))  # Store positions after moving
        
    return positions

# Example usage
N = 4  # Number of MZI modes
positions = move_photons(N)

print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")
#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_clements_scheme(N):
    # Define parameters
    top_divide = N // 2 + 1
    bottom_divide = N // 2 + 2
    
    # Create time bins
    bottom_points = np.linspace(0, 2 * np.pi, bottom_divide)
    top_points = np.linspace(0, 2 * np.pi, top_divide)
    
    # Set up plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_axis_off()
    
    # Plot bottom loop
    for i in range(N // 2 + 1):
        ax.plot([bottom_points[i], bottom_points[i+1]], [0, 1], color='brown')
        
    # Plot top loop
    for i in range(N // 2):
        ax.plot([top_points[i], top_points[i+1]], [2, 3], color='blue')
    
    # Plot interstitial bin
    ax.plot(bottom_points[1], 1, 'ro')  # Red dot for the interstitial bin
    
    # Annotate modes with their labels
    for i in range(1, N+1):
        if i % 2 == 0:
            ax.text(bottom_points[1], 0, str(i), ha='center', va='bottom')  # Even modes in the bottom loop
        else:
            ax.text(bottom_points[1], 2, str(i), ha='center', va='bottom')  # Odd modes in the top loop
    
    # Add title and show plot
    
    plt.show()

# Example usage
N = 4  # Number of MZI modes
plot_clements_scheme(N)

import numpy as np

def move_photons(N):
    top_divide = N // 2 + 1
    bottom_divide = N // 2 + 2
    bottom_points = np.linspace(0, 2 * np.pi, bottom_divide)
    top_points = np.linspace(0, 2 * np.pi, top_divide)
    
    positions = []
    photon_positions = []  # Initialize an empty list to store positions
    
    # Introduce each photon with corresponding label indicating the loop it belongs to
    for i in range(1, N + 1):
        if i % 2 == 0:
            photon_positions.append((f'bottom', bottom_points[1]))  # Introduced as bottom loop mode
        else:
            photon_positions.append((f'top', bottom_points[1]))     # Introduced as top loop mode
            
    positions.append(list(photon_positions))  # Store positions after introducing all photons
    
    # Move all photons up by 1 at each step while alternating interference patterns
    for _ in range(1, N):
        # Alternate interference patterns between odd and even modes
        if _ % 2 == 0:
            photon_positions = [(label, angle + 1) for label, angle in photon_positions]
        else:
            photon_positions[0] = (photon_positions[0][0], bottom_points[1])  # Move the first mode to the interstitial bin
            photon_positions = [(photon_positions[-1][0], bottom_points[1])] + photon_positions[:-1]  # Rotate modes
            
        positions.append(list(photon_positions))  # Store positions after moving
        
    return positions

# Example usage
N = 4  # Number of MZI modes
positions = move_photons(N)

print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")

# %%
import numpy as np

def move_photons(N):
    # Define parameters for circles
    top_divide = N // 2 + 1
    bottom_divide = N // 2 + 2
    
    # Create angles for bottom and top points
    bottom_points = np.linspace(0, 2 * np.pi, bottom_divide)
    top_points = np.linspace(0, 2 * np.pi, top_divide)
    
    # Initialize positions
    positions = []
    photon_positions = []
    
    # Introduce photons with corresponding labels
    for i in range(1, N + 1):
        if i % 2 == 0:
            photon_positions.append(('bottom', bottom_points[1]))  # Introduced as even, starting from bottom[1]
        else:
            photon_positions.append(('top', bottom_points[1]))     # Introduced as odd, starting from bottom[1]
    
    positions.append(list(photon_positions))  # Store positions after introducing all photons
    
    # Move photons up and switch loops according to Clements scheme
    for _ in range(1, N):
        # Switch loop configuration if needed
        if _ % 2 == 0:
            top_loop = np.arange(N // 2)  # Configure even modes to top loop
            bottom_loop = np.arange(N // 2 + 1)
        else:
            top_loop = np.arange(N // 2 + 1)  # Configure odd modes to top loop
            bottom_loop = np.arange(N // 2)
        
        # Move photons
        for i in range(N):
            if i % 2 == 0:
                photon_positions[i] = ('bottom', bottom_points[i % (N // 2) + 1])  # Move even photons in bottom loop
            else:
                photon_positions[i] = ('top', top_points[i % (N // 2) + 1])         # Move odd photons in top loop
        
        positions.append(list(photon_positions))  # Store positions after moving
        
        # Check for crossover
        bottom_x_points = np.cos(bottom_points + np.pi/2 - bottom_points[1])
        bottom_y_points = np.sin(bottom_points + np.pi/2 - bottom_points[1])
        top_x_points = np.cos(top_points + np.pi/2 - top_points[1])
        top_y_points = np.sin(top_points + np.pi/2 - top_points[1])
        
        if np.allclose(bottom_x_points, top_x_points) and np.allclose(bottom_y_points, top_y_points):
            print("Crossover detected at step", _)
        
    return positions

# Example usage
N = 4  # Number of MZI modes
positions = move_photons(N)

print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")
#%%
import numpy as np

def move_photons(N):
    # Define parameters for circles
    top_divide = N // 2 + 1
    bottom_divide = N // 2 + 2
    
    # Create angles for bottom and top points
    bottom_points = np.linspace(0, 2 * np.pi, bottom_divide)
    top_points = np.linspace(0, 2 * np.pi, bottom_divide)  # Make top_points same length as bottom_points
    
    # Initialize positions
    positions = []
    photon_positions = []
    
    # Introduce photons with corresponding labels
    for i in range(1, N + 1):
        if i % 2 == 0:
            photon_positions.append(('bottom', bottom_points[1]))  # Introduced as even, starting from bottom[1]
        else:
            photon_positions.append(('top', bottom_points[1]))     # Introduced as odd, starting from bottom[1]
    
    positions.append(list(photon_positions))  # Store positions after introducing all photons
    
    # Move photons up and switch loops according to Clements scheme
    for _ in range(1, N):
        # Switch loop configuration if needed
        if _ % 2 == 0:
            top_loop = np.arange(N // 2)  # Configure even modes to top loop
            bottom_loop = np.arange(N // 2 + 1)
        else:
            top_loop = np.arange(N // 2 + 1)  # Configure odd modes to top loop
            bottom_loop = np.arange(N // 2)
        
        # Move photons
        for i in range(N):
            if i % 2 == 0:
                photon_positions[i] = ('bottom', bottom_points[i % (N // 2) + 1])  # Move even photons in bottom loop
            else:
                photon_positions[i] = ('top', top_points[i % (N // 2) + 1])         # Move odd photons in top loop
        
        positions.append(list(photon_positions))  # Store positions after moving
        
        # Check for crossover
        bottom_x_points = np.cos(bottom_points + np.pi/2 - bottom_points[1])
        bottom_y_points = np.sin(bottom_points + np.pi/2 - bottom_points[1])
        top_x_points = np.cos(top_points + np.pi/2 - top_points[1])+1.5
        top_y_points = np.sin(top_points + np.pi/2 - top_points[1])+1.5
        
        if np.allclose(bottom_x_points, top_x_points) and np.allclose(bottom_y_points, top_y_points):
            print("Crossover detected at step", _)
        
    return positions

# Example usage
N = 4  # Number of MZI modes
positions = move_photons(N)

print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")

# %%
#%%
import numpy as np
print("right")

N = 4  # Number of MZI modes
Np=N+1
bottom_divide = int(N / 2 + 2) 
bottom_loop = np.arange(N // 2 +1)
positions = move_photons(Np)
def move_photons(Np):
    N = 4 #number of mzi modes (even only)
    #divide circles into n+1 parts
    top_divide = int(N / 2 + 1) 
    bottom_divide = int(N / 2 + 2) 
    #create n+1 equally spaced angles
    bottom_points = np.linspace(0, 2*np.pi, bottom_divide)  
    top_points = np.linspace(0, 2*np.pi, top_divide)
    #set radii
    bottom_r = 1
    top_r = 0.5
    #rotation algorithm for the roatation angle so the bottom and top circles align at 0,1 
    rotation_bottom = np.pi/2 - bottom_points[1]
    
    rotation_top = np.pi/2 - top_points[1]
    #rotate time-bins for bottom points
    bottom_x_points = bottom_r * np.cos(bottom_points + rotation_bottom)
    bottom_y_points = bottom_r * np.sin(bottom_points + rotation_bottom)
    
    print(bottom_x_points,bottom_y_points)
    #rotate time-bins for top points
    top_x_points = top_r * np.cos(top_points - rotation_top)  
    top_y_points = top_r * np.sin(top_points  - rotation_top)
    print(top_x_points,top_y_points)
    positions = []
    photon_positions = [] 
    
    # input photon w corresponding label 
    for i in range(1, N + 1):
        if i % 2 == 0:
            photon_positions.append((f'bottom', bottom_points[1]))  #even
        else:
            photon_positions.append((f'top', bottom_points[1]))   #odd       
    positions.append(list(photon_positions))  # Store positions after introducing all photons

    if np.allclose(bottom_x_points, top_x_points) and np.allclose(bottom_y_points, top_y_points):
            print("Crossover detected at step", _)



    '''
    # Move all photons up by 1 at each step
    for _ in range(1, N):
        photon_positions = [(label, angle + 1) for label, angle in photon_positions]
        # Adjust positions for each photon to follow the desired pattern
        if _ < N - 1:
            photon_positions[_ + 1] = (photon_positions[_][0], bottom_points[1])
        positions.append(list(photon_positions))  # Store positions after moving
        '''
    return positions
long=int(len(bottom_loop))
print(long)
positions = []
print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")

# %%
import numpy as np

def move_photons(N):
    # Define parameters for circles
    top_divide = N // 2 + 1
    bottom_divide = N // 2 + 2
    
    # Create angles for bottom and top points
    bottom_points = np.linspace(0, 2 * np.pi, bottom_divide)
    top_points = np.linspace(0, 2 * np.pi, bottom_divide)  # Make top_points same length as bottom_points
    
    # Initialize positions
    positions = []
    photon_positions = []
    
    # Introduce photons with corresponding labels
    for i in range(1, N + 1):
        if i % 2 == 0:
            photon_positions.append(('bottom', bottom_points[1]))  # Introduced as even, starting from bottom[1]
        else:
            photon_positions.append(('top', bottom_points[1]))     # Introduced as odd, starting from bottom[1]
    
    positions.append(list(photon_positions))  # Store positions after introducing all photons
    
    # Move photons up and switch loops according to Clements scheme
    for _ in range(1, N):
        # Switch loop configuration if needed
        if _ % 2 == 0:
            top_loop = np.arange(N // 2)  # Configure even modes to top loop
            bottom_loop = np.arange(N // 2 + 1)
        else:
            top_loop = np.arange(N // 2 + 1)  # Configure odd modes to top loop
            bottom_loop = np.arange(N // 2)
        
        # Move photons
        for i in range(N):
            if i % 2 == 0:
                photon_positions[i] = ('bottom', bottom_points[i % (N // 2) + 1])  # Move even photons in bottom loop
            else:
                photon_positions[i] = ('top', top_points[i % (N // 2) + 1])         # Move odd photons in top loop
        
        positions.append(list(photon_positions))  # Store positions after moving
        bottom_r = 1
        top_r = 0.5
        # Check for crossover
        rotation_bottom = np.pi/2 - bottom_points[1]
    

        rotation_top = np.pi/2 - top_points[1]

        #rotate time-bins for bottom points
        bottom_x_points = bottom_r * np.cos(bottom_points + rotation_bottom)
        print("bottom x") 
        print(bottom_x_points)
        bottom_y_points = bottom_r * np.sin(bottom_points + rotation_bottom)
        print("bottom y") 
        print(bottom_y_points)

        
        #rotate time-bins for top points
        top_x_points = top_r * np.cos(top_points - rotation_top)+1.5
        print("tops x") 
        print(top_x_points)

        top_y_points = top_r * np.sin(top_points  - rotation_top)+1.5
        print("tops y") 
        print(top_y_points)

        print("tops")   
        if np.allclose(bottom_x_points, top_x_points) and np.allclose(bottom_y_points, top_y_points):
            print("Crossover detected at step", _)
        
    return positions

# Example usage
N = 4  # Number of MZI modes
positions = move_photons(N)

print("Photon positions at each step:")
for i, step in enumerate(positions, 1):
    print(f"Step {i}:")
    for j, (label, angle) in enumerate(step, 1):
        print(f"  Photon {j}: Label = {label}, Angle = {angle}")

# %%
#how many timebins in each loop
import numpy as np
N=4
top_divide = N // 2 
bottom_divide = N // 2 + 1

current_mode = 1
top_array=np.empty(top_divide)
bottom_array=np.empty(bottom_divide)
timebins=N*len(bottom_loop)
for i in range(timebins):
    bottom_index = i % bottom_divide
    bottom_array[bottom_index] = current_mode
    
    if bottom_index < top_divide:
        top_array[bottom_index] = current_mode
    
    current_mode += 1

print("Bottom Loop:")
print(bottom_array)

print("Top Loop:")
print(top_array)


# %%
import numpy as np #this code cycles properly in the bottom loop dont change it

N = 4
top_divide = N // 2 
bottom_divide = N // 2 + 1
top_array=np.empty(top_divide, dtype=object)
bottom_array = np.empty(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide

for i in range(timebins):
    bottom_index = i % bottom_divide
    
    # Shift the elements to the right
    for j in range(bottom_divide - 1, 0, -1):
        bottom_array[j] = bottom_array[j - 1]
    
    bottom_array[0] = f"Mode {current_mode} (Bottom)"
    
    # Move to the next mode with alternating odd and even numbers
    current_mode += step_size
    
    # Wrap around the modes
    if current_mode > N:
        # Set the next mode to be an even number
        current_mode = 2
    elif current_mode > N - 1:
        # If the next mode exceeds N, set it to 1
        current_mode = 1
    
    print(f"At timebin {i + 1}:")
    print("Bottom Loop:")
    print(bottom_array)
    print()

print("Final state:")
print("Bottom Loop:")
print(bottom_array)

# %%
import numpy as np

N = 4
top_divide = N // 2 
bottom_divide = N // 2 + 1

top_array = np.empty(top_divide, dtype=object)
bottom_array = np.empty(bottom_divide, dtype=object)

# Initialize current_mode
current_mode = 1

timebins = N * bottom_divide

for i in range(timebins):
    bottom_index = i % bottom_divide
    
    # Shift elements in bottom_array to the right
    for j in range(bottom_divide - 1, 0, -1):
        bottom_array[j] = bottom_array[j - 1]
    
    # Add current mode to bottom_array
    bottom_array[0] = f"Mode {current_mode} (Bottom)"

    # Increment current_mode
    current_mode += 1
    
    # Check if current_mode exceeds N, wrap around to 1
    if current_mode > N:
        current_mode = 1
    
    # Update top_array with the previous mode label from bottom_array
    if bottom_index > 0:  # Ignore the first element of bottom_array
        top_array[bottom_index - 1] = bottom_array[bottom_index]
    
    # Insert mode labels into top_array in alternating odd and even fashion
    if current_mode % 2 != 0:
        top_array[0] = bottom_array[bottom_divide // 2]
    else:
        top_array[0] = None
    
    # Print current state
    print(f"At timebin {i + 1}:")
    print("Bottom Loop:")
    print(bottom_array)
    print("Top Loop:")
    print(top_array)
    print()

print("Final state:")
print("Bottom Loop:")
print(bottom_array)
print("Top Loop:")
print(top_array)

# %%

import numpy as np #this code cycles properly in the bottom loop dont change it

N = 4
top_divide = N // 2 
bottom_divide = N // 2 + 1
top_array=np.empty(top_divide, dtype=object)
bottom_array = np.empty(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2
for i in range(timebins):
    bottom_index = i % bottom_divide
    
    # Shift the elements to the right
    for j in range(bottom_divide - 1, 0, -1):
        bottom_array[j] = bottom_array[j - 1]
    
    bottom_array[0] = f"Mode {current_mode} (Bottom)"
    
    # Move to the next mode with alternating odd and even numbers
    current_mode += step_size
    
    # Wrap around the modes
    if current_mode > N:
        # Set the next mode to be an even number
        current_mode = 2
    elif current_mode > N - 1:
        # If the next mode exceeds N, set it to 1
        current_mode = 1
    if bottom_index==middle_index:


    print(f"At timebin {i + 1}:")
    print("Bottom Loop:")
    print(bottom_array)
    print()

print("Final state:")
print("Bottom Loop:")
print(bottom_array)


#%% Working code
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2==0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.empty(top_divide, dtype=object)
bottom_array = np.empty(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2
for i in range(timebins):
    bottom_index = i % bottom_divide
    
    # Shift the elements to the right
    for j in range(bottom_divide - 1, 0, -1):
        bottom_array[j] = bottom_array[j - 1]
    
    bottom_array[0] = f"Mode {sorted_modes} (Bottom)"
    if bottom_index==middle_index and ((current_mode-1) % 2) != 0:
             top_array[0] = f"Mode {current_mode+1}"
    # Move to the next mode with alternating odd and even numbers
    current_mode += step_size
    
    # Wrap around the modes
    if current_mode > N:
        # Set the next mode to be an even number
        current_mode = 2
    elif current_mode > N - 1:
        # If the next mode exceeds N, set it to 1
        current_mode = 1
    print(f"At timebin {i + 1}:")
    print("Bottom Loop:")
    print(bottom_array)
    print("Top Loop:")
    print(top_array)
    print()

print("Final state:")
print("Bottom Loop:")
print(bottom_array)
print("Top Loop:")
print(top_array)


# %%
import numpy as np
N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2==0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.empty(top_divide, dtype=object)
bottom_array = np.empty(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2
for curTopBin, curBotBin in zip(top_array, bottom_array):
  
    # print(array_length)
    middle_index = (array_length - 1) // 2
    # print(middle_index)

    print("Bin")
    for i in range(0, middle_index):
        print(curTopBin[i])

    print()

    for i in range(0, len(curBotBin)):
        print(curBotBin[i])

# %%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print("Sorted Modes:", sorted_modes)

top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.empty(top_divide, dtype=object)
bottom_array = np.empty(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2

for i in range(timebins):
    bottom_index = i % bottom_divide
    
    # Shift the elements in the bottom array to the right
    for j in range(bottom_divide - 1, 0, -1):
        bottom_array[j] = bottom_array[j - 1]
    
    # Set the current mode based on the sorted modes
    current_mode = sorted_modes[i % N]
    bottom_array[0] = f"Mode {current_mode} (Bottom)"
    
    # If the current mode is odd, place it in the top array
    if current_mode % 2 != 0:
        for j in range(top_divide - 1, 0, -1):
            top_array[j] = top_array[j - 1]
        top_array[0] = f"Mode {current_mode} (Top)"
    else:
        top_array[0] = None
    
    print(f"At timebin {i + 1}:")
    print("Bottom Loop:")
    print(bottom_array)
    print("Top Loop:")
    print(top_array)
    print()

print("Final state:")
print("Bottom Loop:")
print(bottom_array)
print("Top Loop:")
print(top_array)



# %%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2==0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2


def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index):
    for i in range(timebins):
        bottom_index = i % bottom_divide
    
    # Shift the elements to the right
    for j in range(bottom_divide - 1, 0, -1):
        bottom_array[j] = bottom_array[j - 1]
        print(bottom_array)
    #if middle_index != 0 :
       # middle()
    return
up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index)


def middle(mode):
    if mode% 2 != 0:
         bottom_array[middle_index] = bottom_array[j - 1]
         top_loop(mode)
         return
    else: mode=0
    return


def top_loop(top_array):
    for j in range(bottom_divide - 1, 0, -1):
        top_array[j] = top_array[j - 1]
    return
# %%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2


def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index):
    for i in range(timebins):
        bottom_index = i % bottom_divide
    
        # Shift the elements to the right
        for j in range(bottom_divide - 1, 0, -1):
            if i == 0:
                bottom_array[j] = sorted_modes[j - 1]
            else:
                bottom_array[j] = bottom_array[j - 1]

        print(f"ts={i + 1}")
        print("bottom array:", bottom_array)

up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index)
#%%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2


def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index):
    for i in range(timebins):
        bottom_index = i % bottom_divide
    
        # Shift the elements to the right
        for j in range(bottom_divide - 1, 0, -1):
            if i == 0:
                bottom_array[j] = sorted_modes[j - 1]
            else:
                bottom_array[j] = bottom_array[j - 1]
        bottom_array[0] = sorted_modes[i % N]

        print(f"ts={i + 1}")
        print("bottom array:", bottom_array)

up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index)
#%%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2

count=0

def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index,count,top_divide ):
    for i in range(timebins):
        top_divide=top_divide
        bottom_index = i % bottom_divide
        
        # Move elements to the right in the bottom array
        for j in range(bottom_divide - 1, 0, -1):
            bottom_array[j] = bottom_array[j - 1]
        
        middle_value = bottom_array[middle_index]
        if middle_value !=0 and middle_value %2 !=0:
            middle(middle_value,top_divide,count)

            middle_value=0
            
        # Add the current mode to the bottom array
        bottom_array[0] = sorted_modes[i % N]

        print(f"ts={i + 1}")
        print("bottom array:", bottom_array)

up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index,count,top_divide)

def middle(mode,top_divide,count):
    count+=1
    if mode% 2 != 0:
         for j in range(top_divide - 1, 0, -1):
            top_array[j] = top_array[j - 1]
         print(mode)
         return
    else: mode=0
    return
# %%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2


def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, count, top_divide):
    for i in range(timebins):
        bottom_index = i % bottom_divide
        
        # Move elements to the right in the bottom array
        for j in range(bottom_divide - 1, 0, -1):
            bottom_array[j] = bottom_array[j - 1]
        
        middle_value = bottom_array[middle_index]
        if middle_value != 0 and middle_value % 2 != 0:
            count = middle(middle_value, top_divide, count)
            middle_value = 0  # Reset middle_value to 0
            
        # Add the current mode to the bottom array
        bottom_array[0] = sorted_modes[i % N]

        print(f"ts={i + 1}")
        print("bottom array:", bottom_array)


def middle(mode, top_divide, count):
    if mode == 0:  # Skip operation if mode is 0
        return count
    count += 1
    if mode % 2 != 0:
        for j in range(top_divide - 1, 0, -1):
            top_array[j] = top_array[j - 1]
        print(mode)
    else:
        mode = 0

    return count


up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index, 0, top_divide)

# %%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2


def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index):
    count = 0
    for i in range(timebins):
        bottom_index = i % bottom_divide
        
        # Move elements to the right in the bottom array
        for j in range(bottom_divide - 1, 0, -1):
            bottom_array[j] = bottom_array[j - 1]
        
        middle_value = bottom_array[middle_index]
        if middle_value != 0 and middle_value % 2 != 0:
            count = middle(middle_value, top_divide, count)
        bottom_array[0] = 0  # Reset bottom array to 0
        
        print(f"ts={i + 1}")
        print("bottom array:", bottom_array)
        print("top array:", top_array)

    return


def middle(mode, top_divide, count):
    count += 1
    if mode % 2 != 0:
        # Move elements to the right in the top array
        for j in range(top_divide - 1, 0, -1):
            top_array[j] = top_array[j - 1]
        top_array[0] = mode  # Add the middle value to the first element of the top array
        print(mode)
    return count


up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index)
#%%
import numpy as np

N = 4
modes = list(range(1, N + 1))

sorted_modes = sorted(modes, key=lambda x: (x % 2 == 0, x))
print(sorted_modes)
top_divide = N // 2
bottom_divide = N // 2 + 1
top_array = np.zeros(top_divide, dtype=object)
bottom_array = np.zeros(bottom_divide, dtype=object)

# Initialize current_mode with the first odd number
current_mode = 1

# Set the step size to alternate between odd and even numbers
step_size = 2

timebins = N * bottom_divide
array_length = len(bottom_array)
middle_index = (array_length - 1) // 2


def up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index):
    count = 0
    for i in range(timebins):
        bottom_index = i % bottom_divide

        # Move elements to the right in the bottom array
        for j in range(bottom_divide - 1, 0, -1):
            bottom_array[j] = bottom_array[j - 1]

        middle_value = bottom_array[middle_index]
        if middle_value != 0 and middle_value % 2 != 0:
            count = middle(middle_value, top_array, count)
            bottom_array[middle_value] = 0  # Reset bottom array to 0
        else:
            bottom_array[middle_value] = sorted_modes[]

        print(f"ts={i + 1}")
        print("bottom array:", bottom_array)
        print("top array:", top_array)

    return


def middle(mode, top_array, count):
    count += 1
    if mode % 2 != 0:
        # Move elements to the right in the top array
        for j in range(len(top_array) - 1, 0, -1):
            top_array[j] = top_array[j - 1]
        top_array[0] = mode  # Add the middle value to the first element of the top array
        print(mode)
    return count


up_to_middle(N, sorted_modes, top_array, bottom_array, timebins, middle_index)

# %%

print('Step \t Time-Bin \t Theta \t Phi')
for step in steps:
    print(f'{step:d} \t 4 \t theta \t phi \t')
