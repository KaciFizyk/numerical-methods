# metoda różnic skończonych

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

dx = 0.5 * 1e-9
w = 20 * dx
x_A = 110 * dx
x_B = 230 * dx
x = np.arange(0, 340 * dx + dx, dx)

h = 1.0545718e-34
m0 = 9.10938356e-31
m_eff = 0.067 * m0
eV = 1.602176634e-19 
V_max = 10 * eV * 0.001

V_array = np.zeros_like(x)

@njit
def V(x):
    result = np.zeros_like(x)
    for i, xi in enumerate(x):
        if x_A - w/2 <= xi <= x_A + w/2:
            result[i] = V_max * (1 + np.cos(2 * np.pi * (xi - x_A) / w)) / 2
        elif x_B - w/2 <= xi <= x_B + w/2:
            result[i] = V_max * (1 + np.cos(2 * np.pi * (xi - x_B) / w)) / 2
    return result

@njit
def shoot(x, V_array, E):
    N = len(x)
    q = np.sqrt(2*m_eff*E)/h
    psi = np.zeros(N, dtype=np.complex128)
    psi[-1] = 1.0 + 0j # psi(x_ost)
    psi[-2] = np.exp(-1j * q * dx)  # psi(x_ost - dx)

    # od prawej na lewo
    for i in range(N - 2, 0, -1):
        psi[i-1] = (2 - 2 * m_eff / h**2 * (E - V_array[i]) * dx**2) * psi[i] - psi[i + 1]
    return psi, q


def probabilities(x, psi, q):
    x1 = x[0]
    x2 = x[1]
    psi1 = psi[0]
    psi2 = psi[1]
    # A = (psi1 * np.exp(1j * q * x1) - psi2 * np.exp(1j * q * x2)) / ((np.exp(2j * q * x1))**2 - (np.exp(2j * q * x2))**2)
    # B = -1 * (-psi2*np.exp(1j * q * x1) + np.exp(1j * q * x2) * psi1) * (np.exp(1j * q * x1 + 1j * q * x2)) / ((np.exp(2j * q * x1))**2 - (np.exp(2j * q * x2))**2)
    Matrix = np.array([[np.exp(1j*q*x1), np.exp(-1j*q*x1)],
                       [np.exp(1j*q*x2), np.exp(-1j*q*x2)]], dtype=np.complex128)
    Y = np.array([psi1, psi2], dtype=np.complex128)
    A, B = np.linalg.solve(Matrix, Y)

    R = np.abs(B)**2 / np.abs(A)**2
    T = 1 / np.abs(A)**2
    return A, B, R, T

def modul(psi):
    return np.abs(psi)**2


# ---- ZADANIE 1 ----
V_array = V(x)
E = 7 * eV * 0.001


psi, q = shoot(x, V_array, E)


A, B, R, T = probabilities(x, psi, q)

print(f"A: {A:.5f}")
print(f"B: {B:.5f}")
print(f"Prawdopodobieństwo odbicia R = {R:.5f}")
print(f"Prawdopodobieństwo transmisji T = {T:.5f}")
print(f"R + T = {R+T:.3f}")

psi_sq = modul(psi)

# Fala po lewej stronie
mask_left = x < (x_A - w/2)
x_left = x[mask_left]
psi_left = A * np.exp(1j * q * x_left) + B * np.exp(-1j * q * x_left)
psi_left_sq = modul(psi_left)

plt.figure(figsize=(8, 4))
plt.plot(x/1e-9, psi_sq, color = 'b', label=r'$|\Psi(x)|^2$')
plt.plot(x_left/1e-9, psi_left_sq, 'g--', label=r'$|\Psi_{<}(x)|^2$')
plt.plot(x/1e-9, V_array/eV/0.001, color = 'r', linestyle='--', label='V(x)')
plt.legend(loc="upper right")
plt.xlabel("x [nm]")
plt.ylabel("E [meV]")
plt.show()


# ---- ZADANIE 2 ----

E_min = 0.001 * eV * 1e-3
E_max = 50 * eV * 1e-3
E_array = np.linspace(E_min, E_max, 10000)
R_array = []
T_array = []

for E in E_array:
    psi, q = shoot(x, V_array, E)
    A, B, R, T = probabilities(x, psi, q)
    R_array.append(R)
    T_array.append(T)

plt.figure(figsize=(10,5))
plt.plot(E_array/eV/0.001, T_array, 'b', label='T(E) - przejście')
plt.plot(E_array/eV/0.001, R_array, 'r', label='R(E) - odbicie')
plt.xlabel("E [meV]")
plt.ylabel("Prawdopodobieństwo")
plt.title("Prawdopodobieństwo przejścia i odbicia dla podwójnej bariery")
plt.legend()
plt.grid(True)
plt.show()


# ---- ZADANIE 3 ----

R_array = np.array(R_array)
T_array = np.array(T_array)

# Szukamy czterech rezonansów (T ~ 1)
from scipy.signal import find_peaks
peaks, properties = find_peaks(T_array, height=0.9)  # height=0.9 - tylko duże maksima

E_resonances = E_array[peaks]
print("Rezonanse: ", E_resonances)

for i, E_Tmax in enumerate(np.sort(E_resonances)[:4]):
    psi, q = shoot(x, V_array, E_Tmax)
    A, B, R, T = probabilities(x, psi, q)

    print(f"\nRezonans {i+1}")
    print(f"E = {E_Tmax/eV/0.001:.4f} meV")
    print(f"Prawdopodobieństwo odbicia R = {R:.5f}")
    print(f"Prawdopodobieństwo transmisji T = {T:.5f}")
    print(f"Suma R + T = {R+T:.5f}")

    psi_sq = modul(psi)

    plt.figure(figsize=(8, 4))
    plt.plot(x / 1e-9, psi_sq, color='b', label=r'$|\Psi(x)|^2$')
    plt.plot(x / 1e-9, V_array / eV / 0.001, 'r--', label='V(x)')
    plt.title(f'Kwadrat modułu funkcji falowej dla E = {E_Tmax/eV/0.001:.3f} meV')
    plt.legend(loc="upper right")
    plt.xlabel("x [nm]")
    plt.ylabel("|Ψ(x)|²")
    plt.grid(True)
    plt.show()
