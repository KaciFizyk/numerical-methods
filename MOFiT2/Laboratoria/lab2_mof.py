import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Stałe i przeliczniki
L_nm = 100.0
bohr_nm = 0.05291772
L = L_nm / bohr_nm

N = 300
dx = L / N

m_eff = 0.067
Econv_meV_to_au = 27211.6

x = np.linspace(0, L, N+1)
# zerowy potencjał 
V_array = np.zeros(N+1)

@njit
def H_psi(psi, V_array):
    Hpsi = np.zeros_like(psi)
    for i in range(1, len(psi) - 1):
        Hpsi[i] = - (1 / (2 * m_eff)) * (psi[i+1] + psi[i-1] - 2 * psi[i]) / dx**2 + V_array[i] * psi[i]

    return Hpsi

@njit
def normalize(Psi, dx):
    C = np.sum(Psi**2) * dx
    return Psi / np.sqrt(C)

@njit
def expected_energy(psi):
    E = np.sum(psi * H_psi(psi, V_array) * dx)
    return E

@njit
# Metoda iteracji w czasie urojonym; psi_orth to flaga do zadania z stanem wzbudzonym
def imaginary_time_step(psi, max_iter=200000, tol=1e-9, alpha=2, psi_orth=None, V_array=V_array):
    E_tab = []
    for i in range(max_iter):
        Hpsi = H_psi(psi, V_array)
        psi_prim = psi - alpha * Hpsi
        psi_prim = normalize(psi_prim, dx)
        psi_prim[0] = psi_prim[-1] = 0

        if psi_orth is not None:
            psi_prim = ortonormalize(psi_orth, psi_prim, dx)
        
        psi = psi_prim
        exp_E = expected_energy(psi_prim)

        if (i > 0 and abs(E_tab[i-1] - exp_E) < (tol / Econv_meV_to_au)):
            print(i)
            break

        E_tab.append(exp_E)



    return psi, E_tab, i+1

@njit
def ortonormalize(psi1, psi2, dx):
    c1 = np.sum(psi1 * psi2 * dx)
    psi2_ortho = psi2 - c1 * psi1
    return normalize(psi2_ortho, dx)


# -------------- ZADANIE 1.2 ---------------


# Wykres dla jednego alfa = 0.95 * m_eff * dx**2 = 0.95 * beta
beta = m_eff * dx**2
print(f"alpha_max = {beta:.4f}")

alpha095 = 0.95*beta
plt.figure(figsize=(8,5))
psi_N, E_tab, _ = imaginary_time_step(np.random.uniform(-1, 1, size=N+1), max_iter=500000, alpha=alpha095)
plt.plot(x * 0.05292, psi_N, label=f"$\\alpha={alpha095:.2f}$")
plt.xlabel("x [nm]")
plt.ylabel(r"$\Psi(x)$")
plt.show()

# Wykresy zbieżności energii od iteracji dla różnych alfa oraz funkcji falowych (podstawowy stan)
alphas = [0.3, 0.5, 2.6, 2.66]
colors = ['b', 'g', 'r', 'c']

plt.figure(figsize=(8,5))
psi_final_list = []

for i, alpha in enumerate(alphas):
    psi = np.random.uniform(-1, 1, size=N+1)
    psi[0] = psi[-1] = 0
    psi = normalize(psi, dx)
    psi_final, E_tab, n_iter = imaginary_time_step(psi, max_iter=500000, alpha=alpha)
    E_tab = np.array(E_tab) * Econv_meV_to_au

    plt.plot(np.arange(len(E_tab)), E_tab, colors[i % len(colors)], label=f"$\\alpha={alpha:.2f}$")
    psi_final = normalize(psi_final, dx)
    psi_final_list.append(psi_final)

plt.xlabel("Numer iteracji")
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Energia oczekiwana <E> [meV]")
plt.title(r"Zbieżność energii dla różnych $\alpha$")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
for i, alpha in enumerate(alphas):
    plt.plot(x * 0.05292, psi_final_list[i], colors[i % len(colors)], label=f"$\\alpha={alpha:.2f}$")

plt.xlabel("x [nm]")
plt.ylabel(r"$\Psi(x)$")
plt.title(r"Funkcja falowa stanu podstawowego dla różnych $\alpha$")
plt.grid(True)
plt.legend()
plt.show()


# Wykres energii końcowej w funkcji beta (alfa)
betas = np.linspace(0.1*beta, 1.2*beta, 30)
fractions = betas / beta
E_betas = []

for beta in betas:
    np.random.seed(40)
    psi = np.random.uniform(-1, 1, size=N+1)
    psi[0] = psi[-1] = 0
    psi = normalize(psi, dx)

    psi_final, E_tab, n_iter = imaginary_time_step(psi, max_iter=50000, alpha=beta)
    E_betas.append(E_tab[-1] * Econv_meV_to_au)

plt.figure(figsize=(8,5))
plt.plot(fractions, E_betas, 'r-', label="Energia końcowa <E>")
plt.axvline(x=1, color='k', linestyle='--', label=r"$\alpha_{crit}=m \cdot \Delta x^2  /  \beta=1$")
plt.xlabel(r"$\beta$ (ułamek $\alpha$)")
plt.ylabel("Energia końcowa <E> [meV]")
plt.grid(True)
plt.legend()
plt.show()


# -------------- ZADANIE 1.4 ---------------

# Dodanie stanu wzbudzonego (ortonormalizacja)
psi = np.random.uniform(-1, 1, size=N+1)
psi[0] = psi[-1] = 0
psi = normalize(psi, dx)
# stan podstawowy
psi1, E_tab1, _ = imaginary_time_step(psi, max_iter=500000, alpha=alpha095)
# stan wzbudzony pierwszy
psi2, E_tab2, _ = imaginary_time_step(psi, max_iter=500000, alpha=alpha095, psi_orth=psi1)

plt.figure(figsize=(8,5))
plt.plot(x * 0.05292, psi1, 'b-', label=r"$\Psi_1$ — stan podstawowy")
plt.plot(x * 0.05292, psi2, 'r-', label=r"$\Psi_2$ — pierwszy stan wzbudzony")
plt.xlabel("x [nm]")
plt.ylabel(r"$\Psi(x)$")
plt.title("Funkcje falowe: stan podstawowy i pierwszy wzbudzony")
plt.legend()
plt.grid(True)
plt.show()

print(f"E1 - stan podstawowy = {E_tab1[-1]*Econv_meV_to_au:.4f} meV")
print(f"E2 - pierwszy stan wzbudzony = {E_tab2[-1]*Econv_meV_to_au:.4f} meV")

plt.figure(figsize=(8,5))
plt.plot(np.arange(len(E_tab1)), np.array(E_tab1)*Econv_meV_to_au, 'b-', label="<E1> — stan podstawowy")
plt.plot(np.arange(len(E_tab2)), np.array(E_tab2)*Econv_meV_to_au, 'r-', label="<E2> — pierwszy stan wzbudzony")
plt.xlabel("Numer iteracji")
plt.ylabel("Energia oczekiwana <E> [meV]")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# -------------- ZADANIE 1.5 ---------------

# Wprowadzamy potencjał W=500 meV
W = 500 / Econv_meV_to_au
V_array[N//2] = -W
print(V_array)

# Stan podstawowy z potencjałem

psi = np.random.uniform(-1, 1, size=N+1)
psi[0] = psi[-1] = 0
psi = normalize(psi, dx)
# stan podstawowy
psi1, E_tab1, _ = imaginary_time_step(psi, max_iter=500000, alpha=alpha095, V_array=V_array)
# stan wzbudzony pierwszy
psi2, E_tab2, _ = imaginary_time_step(psi, max_iter=500000, alpha=alpha095, psi_orth=psi1, V_array=V_array)

plt.figure(figsize=(8,5))
plt.plot(x * 0.05292, psi1, 'b-', label=r"$\Psi_1$ — stan podstawowy")
plt.plot(x * 0.05292, psi2, 'r-', label=r"$\Psi_2$ — pierwszy stan wzbudzony")
plt.xlabel("x [nm]")
plt.ylabel(r"$\Psi(x)$")
plt.title("Funkcje falowe: stan podstawowy i pierwszy wzbudzony")
plt.legend()
plt.grid(True)
plt.show()

print(f"E1 - stan podstawowy = {E_tab1[-1]*Econv_meV_to_au:.4f} meV")
print(f"E2 - pierwszy stan wzbudzony = {E_tab2[-1]*Econv_meV_to_au:.4f} meV")

plt.figure(figsize=(8,5))
plt.plot(np.arange(len(E_tab1)), np.array(E_tab1)*Econv_meV_to_au, 'b-', label="<E1> — stan podstawowy")
plt.plot(np.arange(len(E_tab2)), np.array(E_tab2)*Econv_meV_to_au, 'r-', label="<E2> — pierwszy stan wzbudzony")
plt.xlabel("Numer iteracji")
plt.ylabel("Energia oczekiwana ⟨E⟩ [meV]")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()