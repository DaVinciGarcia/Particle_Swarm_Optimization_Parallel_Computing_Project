import matplotlib.pyplot as plt

particulas = [1000, 3000, 5000, 7500, 10000, 50000, 100000]
secuencial = [78, 198, 320, 488, 647, 3271, 6420]
paralelo_v0 = [535, 817, 1098, 1442, 1797, 12109, 24197]
paralelo_v1 = [157, 429, 696, 1029, 1361, 6734, 14205]
paralelo_v2 = [88, 93, 96, 100, 103, 206, 333]


fig, ax = plt.subplots()

# Graficar los datos
ax.plot(particulas, secuencial, label='Secuential', marker='o')
#ax.plot(particulas, paralelo_v0, label='CUDA V0', marker='o')
#ax.plot(particulas, paralelo_v1, label='CUDA V1', marker='o')
ax.plot(particulas, paralelo_v2, label='CUDA V2', marker='o')

# Añadir etiquetas y título
ax.set_xlabel('Number of particles')
ax.set_ylabel('Execuction Time (ms)')
ax.set_title('Execution Times of PSO of Different Version')
ax.legend()

ax.grid(True)
plt.show()
