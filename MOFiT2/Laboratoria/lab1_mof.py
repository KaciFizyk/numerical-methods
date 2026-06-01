import numpy as np
import matplotlib.pyplot as plt

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

# Metoda strzałów - shooting method - dla danej energii i potencjału zwrócimy całe Psi oraz Psi_N czyli ostatni element Psi
def shoot(E_au, V_array):
    Psi = np.zeros(N+1)
    Psi[0] = 0
    Psi[1] = 1
    for i in range(1, N):
        Psi[i+1] = (2 - 2 * m_eff * (E_au - V_array[i]) * dx**2) * Psi[i] - Psi[i-1]
    return Psi, Psi[-1]

# Bisekcja z tolerancją - w przedziale E1_meV do E2_meV
def bisection(E1_meV, E2_meV, V_array, tol=1e-6, max_iter=20000):
    E1 = E1_meV / Econv_meV_to_au
    E2 = E2_meV / Econv_meV_to_au

    _, Psi1 = shoot(E1, V_array)
    _, Psi2 = shoot(E2, V_array)

    if Psi1 * Psi2 > 0:
        return None

    for _ in range(max_iter):
        Em = 0.5 * (E1 + E2)
        _, Psym = shoot(Em, V_array)

        if Psi1 * Psym < 0:
            E2, Psi2 = Em, Psym
        else:
            E1, Psi1 = Em, Psym

        if abs(E2 - E1) < tol / Econv_meV_to_au:
            break

    return 0.5 * (E1 + E2) * Econv_meV_to_au

# Funkcja znajdująca zadaną ilość miejsc zerowych (max_levels) - potem się przyda
def find_zeros(V_array, E_min=0, E_max=35, N_scan=2000, max_levels=7):
    E_scan = np.linspace(E_min, E_max, N_scan)
    Psi_end = [shoot(E/Econv_meV_to_au, V_array)[1] for E in E_scan]
    zeros = []

    for i in range(len(Psi_end)-1):
        if Psi_end[i] * Psi_end[i+1] < 0:
            E_zero = bisection(E_scan[i], E_scan[i+1], V_array)
            # jak znajdzie zero to zapisz
            if E_zero is not None:
                zeros.append(E_zero)
                # ograniczenie ilości szukanych zer
                if len(zeros) >= max_levels:
                    break
    return zeros

# Funkcja do normalizacji funkcji falowej
def normalize(Psi, dx):
    C = np.sum(Psi**2) * dx
    return Psi / np.sqrt(C)

# Funkcja plotująca funkcję falową dla zadanej energii +- 5% - po prostu 3 wykresy na jednym
def plot_wavefunctions_on_one(E_meV):

    energies_meV = [0.95 * E_meV, E_meV, 1.05 * E_meV]
    labels = [r"$E - 5\%$", r"$E$", r"$E + 5\%$"]
    colors = ["magenta", "black", "purple"]

    plt.figure(figsize=(8, 6))
    
    for E, label, color in zip(energies_meV, labels, colors):
        E_au = E / Econv_meV_to_au
        Psi, _ = shoot(E_au, V_array)
        Psi_norm = normalize(Psi, dx)
        plt.plot(x * 0.05292, Psi_norm, label=label, color=color)

    plt.xlabel("x [nm]")
    plt.ylabel(r"$\Psi(x)$")
    plt.title(f"Funkcje falowe w okolicach E = {E_meV:.4f} meV")
    plt.axhline(0, linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()

# ----------------- ZADANIE 1 ---------------

E_values_meV = np.linspace(0, 35, 500)
Psi_end = [shoot(E/Econv_meV_to_au, V_array)[1] for E in E_values_meV]

plt.figure(figsize=(8,6))
plt.plot(E_values_meV, Psi_end, label=r"$\Psi_N(E)$")
plt.axhline(0, linestyle='--', label="y=0")
plt.xlabel("E [meV]")
plt.ylabel(r"$\Psi(L)$")
plt.title(r"$\Psi(L)$ w funkcji energii E")
plt.legend()
plt.grid()
plt.show()

# dla pierwszego miejsca zerowego z przedzialu 0-1
E1 = bisection(0, 1, V_array)
plot_wavefunctions_on_one(E1)

# ----------------- ZADANIE 2 ---------------

E_levels = find_zeros(V_array, E_min=0, E_max=35, N_scan=2000)
print("Znalezione energie własne (meV):")
for i, E in enumerate(E_levels, start=1):
    print(f"E{i} = {E:.6f} meV")

print("\nWyliczone energie własne (meV) + względny błąd:")
for i, E_num in enumerate(E_levels, start=1):
    E_exact_au = (i**2 * np.pi**2) / (2 * m_eff * L**2)
    E_exact_meV = E_exact_au * Econv_meV_to_au
    print(f"E{i}= {E_exact_meV:.6f} meV, względny błąd = {np.abs(E_num-E_exact_meV)/E_exact_meV*100:.3} %")


# ----------------- ZADANIE 3 ---------------

# Znalezienie miejsc zerowych w funkcji W 
W_values = np.linspace(0, 1, 50)
all_levels = []

for W in W_values:
    # nadajemy niezerowy potencjał na środku
    V = np.zeros(N+1)
    V[N//2] = -W * 1000 / Econv_meV_to_au
    levels = find_zeros(V, E_min=-50, E_max=35, N_scan=2000, max_levels=7)
    all_levels.append(levels)

all_levels = np.array(all_levels)

plt.figure(figsize=(8,6))
for n in range(7):
    plt.plot(W_values, all_levels[:, n], label=f"E{n+1}")
plt.xlabel("W [eV]")
plt.ylabel("E_n [meV]")
plt.title("7 najniższych energii własnych w funkcji bariery -W")
plt.grid(True)
plt.legend()
plt.show()

# Funkcje własne 4 najniższych stanów dla W = 0.5
W = 0.5
V = np.zeros(N+1)
V[N//2] = -W * 1000 / Econv_meV_to_au

levels = find_zeros(V_array=V, E_min=-50, E_max=35, N_scan=2000, max_levels=4)
print("4 najniższe energie własne (meV) dla W = 0.5 eV:")
for i, E in enumerate(levels, start=1):
    print(f"E{i} = {E:.6f} meV")

def normalize(Psi, dx):
    C = np.sum(Psi**2) * dx
    return Psi / np.sqrt(C)

plt.figure(figsize=(8,5))
colors = ["blue", "red", "green", "orange"]
for i, E in enumerate(levels):
    E_au = E / Econv_meV_to_au
    Psi, _ = shoot(E_au, V)
    Psi_norm = normalize(Psi, dx)
    plt.plot(x * bohr_nm, Psi_norm, label=f"E{i+1} = {E:.3f} meV", color=colors[i])

plt.xlabel("x [nm]")
plt.ylabel(r"$\Psi(x)$")
plt.title("4 najniższe funkcje falowe dla W = 0.5 eV")
plt.axvline(L_nm/2, color='k', linestyle='--', label="bariera w środku")
plt.legend()
plt.grid(True)
plt.show()




# Funkcje falowe dla 2 najniżej energetycznych stanów w  zależności od W
W_values = np.linspace(0, 1, 10)
colors = plt.cm.viridis(np.linspace(0,1,len(W_values)))

plt.figure(figsize=(12,5))

for i_state in range(2):  # dla dwóch najniższych stanów
    plt.subplot(1, 2, i_state+1)
    for W, color in zip(W_values, colors):
        V = np.zeros(N+1)
        V[N//2] = -W * 1000 / Econv_meV_to_au
        levels = find_zeros(V, E_min=-50, E_max=35, max_levels=2)
        E_au = levels[i_state] / Econv_meV_to_au
        Psi, _ = shoot(E_au, V)
        Psi_norm = normalize(Psi, dx)
        plt.plot(x*bohr_nm, Psi_norm, color=color, label=f"W={W:.2f} eV")
    
    plt.title(f"Stan {i_state+1}")
    plt.xlabel("x [nm]")
    plt.ylabel(r"$\Psi(x)$")
    plt.axvline(L_nm/2, color='k', linestyle='--', label="bariera")
    plt.grid()
    plt.legend()
        
plt.tight_layout()
plt.show()
