#im sorry code gods
from quotonic.clements import Mesh
import numpy as np
import quotonic.aa as aa
from quotonic.fock import basis, getDim
from quotonic.misc import genHaarUnitary
import math as m
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np
x = PrettyTable()
# mesh size
N = 4
#CHANGE PHOTON TO MODE
# phase changes for pi stuff
phi = 0
photons=np.arange(N+1)
print(photons)
# loops here
top_loop= np.arange(N // 2) #the spots... i see the vision
top_bins=len(top_loop)

bottom_loop = np.arange(N // 2 +1)
print(bottom_loop)


timebins=N*len(bottom_loop)
bottom_bins=len(bottom_loop)
print(bottom_bins)

#for i in 
  #  if photon[i]




#heres to it working... cheers
for tau in range(timebins+1):
    
    print(f"Timebin {tau}:") #at timebin ....

 #   for i in range(len(bottom_loop) - 1):



   #odd modes
    '''for i in range(len(top_loop) - 1):
        
        print(f"Interfere mode {top_loop[i]+1} with mode {top_loop[i+1]+1}")
        phi += np.pi  # Add pi phase shift
        
    # Switch loop configuration
    if tau == 0:
        top_loop = np.concatenate(([N // 2], top_loop))
        bottom_loop = np.delete(bottom_loop, 0)
    else:
        top_loop, bottom_loop = bottom_loop, top_loop
    
    # Interfere even modes
    for i in range(len(top_loop) - 1):
        print(f"Interfere mode {top_loop[i]} with mode {top_loop[i+1]}")
        phi += np.pi  # Add pi phase shift
'''

def circlesplot(): #the circles are done... just animate
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
    #nake circles 
    angles = np.linspace(0, 2*np.pi, 1000) 
    bottom_x_circle = bottom_r * np.cos(angles)
    bottom_y_circle = bottom_r * np.sin(angles)
    top_x_circle = top_r * np.cos(angles) 
    top_y_circle = top_r * np.sin(angles) 
    #plot the plots 
    plt.plot(bottom_x_circle, bottom_y_circle, color='purple')
    plt.plot(bottom_x_points, bottom_y_points, 'D', color='black', label='timebins')
    plt.plot(top_x_circle, top_y_circle+1.5, color='orange')
    plt.plot(top_x_points, top_y_points+1.5, 'D', color='black')
    plt.legend()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    print(top_divide)
    print(bottom_divide)
    print(top_divide+bottom_divide)
    #appreciating the code gods for letting the circles plot right <3
circlesplot()
#%%
''' old code ! but were gonna keep it bc of the diagram shtuff
def loops(timebins): #turn this into mesh size somehow ? ask jacob if hes got any ideas
    timebin=0 #set to 0 
    timebins=timebins
    currentbin=0 #set to 0
    phi= 0 #set to 0 for now since Im not sure what to put them at yet
    theta=0
    #timebins=t*meshsize #think about how you want to implement the mesh size?
    table.add_row([timebin,"off", phi, theta, "on", "inner loop" ])
    timebin+=1
    table.add_row([timebin,"off", phi, theta, "on", "outer loop" ])
    #print(table) #OH MAN IT PRINTS SMALL WINS
    for  i in range (0,timebins): #theres gotta be a better way to keep track of both loops.... think of this a little bit more
        if i%2==0:
            timebin+=1
            table.add_row([timebin,"off", phi, theta, "on", "outer loop" ])
        if i%2==1:
            timebin+=1
            table.add_row([timebin,"on", phi, theta, "off", "inner loop" ])
        if i == timebins-2:
         break
    timebin+=1
    table.add_row([timebin,"on", phi, theta, "on", "outer loop" ]) #last pulse needs to couple to outerloop and then escape thru inner loop
    timebin+=1
    table.add_row([timebin,"on", phi, theta, "off", "inner loop" ]) #last pulse needs to couple to innerloop to escape
    print(table)
loops(9)



def plot_loop(): #right now its just a plot, turn into animation later.... surprised it plotted fr fr
    fig, ax = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0.5, 0.2), 0.2, color='black', fill=False) #inner loop
    circle2 = plt.Circle((0.5, 0.2), 0.4, color='black', fill=False) #outer loop 
    ax.add_artist(circle)
    ax.add_artist(circle2)
    ax.axhline(y=0, color='black', linestyle='-', )
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    ax.axis('off')
    plt.show()
#plot_loop()
'''



'''
#define input photon, need to define in fock state?


def photonsInitial():
    n=2 #lets say 2 photons
    m=3 #3 modes for now
    fockBasis=basis(n,m) #[[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]] ; aight
    return
photonsInitial()

def timeBin(): #source's repetition rate
    #phases ? 
    


'''
