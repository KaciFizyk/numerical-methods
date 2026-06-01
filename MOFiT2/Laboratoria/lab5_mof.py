import numpy as np
import matplotlib.pyplot as plt
from numba import njit

N = 200
delta_rs = np.linspace(0.01, 0.99, 100)

@njit
def energy_levels(N, delta_r, l):
    r = np.linspace(delta_r, N * delta_r, N)
    W = 0.5 * l * (l + 1) / (r ** 2) - 1 / r

    # Stworzenie macierzy Hamiltona
    H = np.zeros((N, N))

    for i in range(N):
        H[i, i] = 1 / delta_r**2 + W[i]
        if i > 0:
            H[i, i - 1] = -1 / (2 * delta_r**2)
        if i < N - 1:
            H[i, i + 1] = -1 / (2 * delta_r**2)

    # Rozwiazanie problemu wlasnego macierzy Hamiltona
    E, psi = np.linalg.eigh(H)
    return E, psi, r



# Zadanie 2

E1_l0 = []
E2_l0, E2_l1 = [], []
E3_l0, E3_l1, E3_l2 = [], [], []

for dr in delta_rs:
    # l=0
    E, _, _ = energy_levels(N, dr, 0)
    E1_l0.append(E[0])      # n=1
    E2_l0.append(E[1])      # n=2
    E3_l0.append(E[2])      # n=3

    # l=1
    E, _, _ = energy_levels(N, dr, 1)
    E2_l1.append(E[0])      # n=2
    E3_l1.append(E[1])      # n=3

    # l=2
    E, _, _ = energy_levels(N, dr, 2)
    E3_l2.append(E[0])      # n=3


E_analytic = lambda n: -1 / (2 * n**2)

# n = 1
plt.figure(figsize=(8, 6))
plt.plot(delta_rs, E1_l0, '-', label='l=0')
plt.axhline(E_analytic(1), color='r', linestyle='--', label='analityczne')
plt.xlabel(r'$\Delta r$')
plt.ylabel(r'$E\,[a.u.]$')
plt.ylim(-0.52, -0.15)
plt.legend()
plt.title(r'Energia dla $n=1$')
plt.grid(True)
# plt.savefig("zad1_n1.png")
plt.show()

# n = 2
plt.figure(figsize=(8, 6))
plt.plot(delta_rs, E2_l0, '-', label='l=0')
plt.plot(delta_rs, E2_l1, '-', label='l=1')
plt.axhline(E_analytic(2), color='r', linestyle='--', label='analityczne')
plt.xlabel(r'$\Delta r$')
plt.ylabel(r'$E\,[a.u.]$')
plt.ylim(-0.14, -0.06)
plt.legend()
plt.title(r'Energie dla $n=2$')
plt.grid(True)
# plt.savefig("zad1_n2.png")
plt.show()

# n = 3
plt.figure(figsize=(8, 6))
plt.plot(delta_rs, E3_l0, '-', label='l=0')
plt.plot(delta_rs, E3_l1, '-', label='l=1')
plt.plot(delta_rs, E3_l2, '-', label='l=2')
plt.axhline(E_analytic(3), color='r', linestyle='--', label='analityczne')
plt.xlabel(r'$\Delta r$')
plt.ylabel(r'$E\,[a.u.]$')
plt.ylim(-0.060, -0.03)
plt.legend()
plt.title(r'Energie dla $n=3$')
plt.grid(True)
# plt.savefig("zad1_n3.png")
plt.show()


# Zadanie 3
delta_r = 0.1
E, psi, r = energy_levels(N, delta_r, 0)

normalize = lambda psi, dr: psi / np.sqrt(np.sum(np.abs(psi)**2) * dr)

psi[:, 0] = normalize(psi[:, 0], delta_r)
psi[:, 1] = normalize(psi[:, 1], delta_r)
psi[:, 2] = normalize(psi[:, 2], delta_r)

plt.figure(figsize=(8,6))
plt.plot(r, psi[:, 0], label=fr'$n=1,\,E={E[0]:.4f}$')
plt.plot(r, psi[:, 1], label=fr'$n=2,\,E={E[1]:.4f}$')
plt.plot(r, psi[:, 2], label=fr'$n=3,\,E={E[2]:.4f}$')
plt.xlabel(r'$r\,[a.u.]$')
plt.ylabel(r'$\varphi_{n,0}(r)$')
plt.title(r'Pomocnicze funkcje falowe $\varphi_{n,l=0}(r)$ dla $\Delta r=0.1$')
plt.legend()
plt.grid(True)
# plt.savefig("zad2.png")
plt.show()

# Zadanie 4
psi_radial = lambda phi, r: phi / r

psi[:,0] = psi_radial(psi[:, 0], r)
psi[:,1] = psi_radial(psi[:, 1], r)
psi[:,2] = psi_radial(psi[:, 2], r)

plt.figure(figsize=(8,6))
plt.plot(r, psi[:,0], label=fr'$n=1,\,E={E[0]:.4f}$')
plt.plot(r, psi[:,1], label=fr'$n=2,\,E={E[1]:.4f}$')
plt.plot(r, psi[:,2], label=fr'$n=3,\,E={E[2]:.4f}$')
plt.xlabel(r'$r\,[a.u.]$')
plt.ylabel(r'$\psi_{n,0}(r)$')
plt.title(r'Funkcje falowe $\psi_{n,l=0}(r)=\varphi_{n,l=0}(r)/r$ dla $\Delta r=0.1$')
plt.legend()
plt.grid(True)
# plt.savefig("zad3.png")
plt.show()