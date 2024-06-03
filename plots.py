import numpy as np
import matplotlib.pyplot as plt

#for tj of .1%
total_length_1=[1.0519391964979217, 1.4727148750970904, 1.8934905536962594,2.7350419108945965,3.5765932680929344,4.418144625291271]
mesh_size_1=[4,6,8,12,16,20]
#for tj of 10%
total_length_10=[1.1438562136676431,1.6013986991347005,2.058941184601758,2.9740261555358725, 3.889111126469987,4.804196097404102]
mesh_size_1=[4,6,8,12,16,20]
#for tj of 50%
total_length_50=[1.552376289977516,2.1733268059685225,2.7942773219595285, 4.036178353941541,5.278079385923554,6.5199804179055665]
mesh_size_1=[4,6,8,12,16,20]

'''plt.plot(mesh_size_1,total_length_1, color='purple', label='tj=1%')
plt.plot(mesh_size_1,total_length_10, label='tj=10%')
plt.plot(mesh_size_1,total_length_50, label='tj=50%')
plt.legend()
plt.xlabel("Mesh Size (NxN)")
plt.ylabel("Fibre Optic Length (m)")
plt.show()'''

# loss plots !! 
loss_avg_50 = [0.09849335458965713,0.13911461736182884,0.1779055142708017,0.21494852033001788, 0.2503223942861451, 0.31636019671102755,0.3765808434467586,0.504935084103102,0.6068627850529142,0.6878048417219306,0.7520819369252866,0.8031251786936019,0.8436592526428967,0.8758478655524078,0.9014092439209983]
mesh_size_1 = [4,6,8,10,12,16,20,30, 40, 50,60,70,80,90,100]

#all of these for 0.045 mzi loss 
# 1-loss
loss_results50 = {}
with open("avgLoss05.txt", "r") as file:
    for line in file:
        parts50 = line.split(":")
        N50 = int(parts50[0].split("=")[1])
        loss50 = float(parts50[1].split("=")[1])
        loss_results50[N50] = loss50

# Extract N values and corresponding losses
N_values50 = list(loss_results50.keys())
loss_values50 = list(loss_results50.values())

loss_results10 = {}
with open("avgLoss01.txt", "r") as file:
    for line in file:
        parts10 = line.split(":")
        N10 = int(parts10[0].split("=")[1])
        loss10 = float(parts10[1].split("=")[1])
        loss_results10[N10] = loss10

# Extract N values and corresponding losses
N_values10 = list(loss_results10.keys())
loss_values10 = list(loss_results10.values())

loss_results01 = {}
with open("avgLoss001.txt", "r") as file:
    for line in file:
        parts01 = line.split(":")
        N01 = int(parts01[0].split("=")[1])
        loss01 = float(parts01[1].split("=")[1])
        loss_results01[N01] = loss01

# Extract N values and corresponding losses
N_values01 = list(loss_results01.keys())
loss_values01 = list(loss_results01.values())




# Plot the results
#plt.plot(N_values0p,loss_values0p, color='cyan', label='Perfect')
#plt.plot(N_values0,loss_values0, color='pink', label='Tj=0')
plt.plot(N_values01,loss_values01, color='purple', label='Tj=1%')
plt.plot(N_values10, loss_values10, linestyle='-', label='Tj=10%')
plt.plot(N_values50, loss_values50, linestyle='-', label='Tj=50%')
#plt.xscale('log')  # Set x-axis to logarithmic scale
#plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Percent Photon Loss vs. Number of Modes')
plt.xlabel("Mesh Size (NxN)")
plt.ylabel("Loss (%)")
plt.legend()
plt.show()


loss_results50 = {}
with open("totalFibreNeeded05.txt", "r") as file:
    for line in file:
        parts50 = line.split(":")
        N50 = int(parts50[0].split("=")[1])
        loss50 = float(parts50[1].split("=")[1])
        loss_results50[N50] = loss50

# Extract N values and corresponding losses
N_values50 = list(loss_results50.keys())
loss_values50 = list(loss_results50.values())

loss_results10 = {}
with open("totalFibreNeeded01.txt", "r") as file:
    for line in file:
        parts10 = line.split(":")
        N10 = int(parts10[0].split("=")[1])
        loss10 = float(parts10[1].split("=")[1])
        loss_results10[N10] = loss10

# Extract N values and corresponding losses
N_values10 = list(loss_results10.keys())
loss_values10 = list(loss_results10.values())

loss_results01 = {}
with open("totalFibreNeeded001.txt", "r") as file:
    for line in file:
        parts01 = line.split(":")
        N01 = int(parts01[0].split("=")[1])
        loss01 = float(parts01[1].split("=")[1])
        loss_results01[N01] = loss01

# Extract N values and corresponding losses
N_values01 = list(loss_results01.keys())
loss_values01 = list(loss_results01.values())




# Plot the results
#plt.plot(N_values0p,loss_values0p, color='cyan', label='Perfect')
#plt.plot(N_values0,loss_values0, color='pink', label='Tj=0')
plt.plot(N_values01,loss_values01, color='purple', label='Tj=1%')
plt.plot(N_values10, loss_values10, linestyle='-', label='Tj=10%')
plt.plot(N_values50, loss_values50, linestyle='-', label='Tj=50%')
#plt.xscale('log')  # Set x-axis to logarithmic scale
#plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Total Fibre Length vs. Number of Modes')
plt.xlabel("Mesh Size (NxN)")
plt.ylabel("Length (m)")
plt.legend()
plt.show()

loss_results50 = {}
with open("avgLoss05HighMZI.txt", "r") as file:
    for line in file:
        parts50 = line.split(":")
        N50 = int(parts50[0].split("=")[1])
        loss50 = float(parts50[1].split("=")[1])
        loss_results50[N50] = loss50

# Extract N values and corresponding losses
N_values50 = list(loss_results50.keys())
loss_values50 = list(loss_results50.values())

loss_results10 = {}
with open("avgLoss01HighMZI.txt", "r") as file:
    for line in file:
        parts10 = line.split(":")
        N10 = int(parts10[0].split("=")[1])
        loss10 = float(parts10[1].split("=")[1])
        loss_results10[N10] = loss10

# Extract N values and corresponding losses
N_values10 = list(loss_results10.keys())
loss_values10 = list(loss_results10.values())

loss_results01 = {}
with open("avgLoss001HighMZI.txt", "r") as file:
    for line in file:
        parts01 = line.split(":")
        N01 = int(parts01[0].split("=")[1])
        loss01 = float(parts01[1].split("=")[1])
        loss_results01[N01] = loss01

# Extract N values and corresponding losses
N_values01 = list(loss_results01.keys())
loss_values01 = list(loss_results01.values())




# Plot the results
#plt.plot(N_values0p,loss_values0p, color='cyan', label='Perfect')
#plt.plot(N_values0,loss_values0, color='pink', label='Tj=0')
plt.plot(N_values01,loss_values01, color='purple', label='Tj=1%')
plt.plot(N_values10, loss_values10, linestyle='-', label='Tj=10%')
plt.plot(N_values50, loss_values50, linestyle='-', label='Tj=50%')
#plt.xscale('log')  # Set x-axis to logarithmic scale
#plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Percent Photon Loss vs. Number of Modes')
plt.xlabel("Mesh Size (NxN)")
plt.ylabel("Loss (%)")
plt.legend()
plt.show() 
