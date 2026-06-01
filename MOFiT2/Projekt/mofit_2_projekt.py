import numpy as np
from numba import njit
from scipy.linalg import eigh, solve
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd

m_eff = 0.067                                   # Masa efektywna m = 0.067 m0
hbar = 1.0                                      # Zredukowana stała Plancka [J*s]
nm_to_bohr = 0.0529177249

hbar_omega = 10 # [MeV]
omega = hbar_omega / hbar / 27211.6             # Częstość kątowa omega [1/s]
L = 100   

# 1. Wczytanie współrzędnych węzłów globalnych
# Plik: wezly_N_2_L_100.dat[Globalny ID], [x (nm)], [y (nm)]
nodes_df2 = pd.read_csv(r'C:\Users\kacpe\Desktop\Kody Warte Uwagi\MOFiT2\Projekt\wezly_N_2_L100.dat', sep='\s+', header=None, names=['ID', 'x_nm', 'y_nm'])
nodes_df2['x'] = nodes_df2['x_nm'] / nm_to_bohr
nodes_df2['y'] = nodes_df2['y_nm'] / nm_to_bohr
idx_border2 = nodes_df2[(np.abs(nodes_df2['x_nm']) == 50) | (np.abs(nodes_df2['y_nm']) == 50)]['ID'].to_numpy()
nodes_map2 = nodes_df2.set_index('ID')[['x', 'y']].to_dict('index')
nodes_df4 = pd.read_csv(r'C:\Users\kacpe\Desktop\Kody Warte Uwagi\MOFiT2\Projekt\wezly_N_4_L100.dat', sep='\s+', header=None, names=['ID', 'x_nm', 'y_nm'])
nodes_df4['x'] = nodes_df4['x_nm'] / nm_to_bohr
nodes_df4['y'] = nodes_df4['y_nm'] / nm_to_bohr
idx_border4 = nodes_df4[(np.abs(nodes_df4['x_nm']) == 50) | (np.abs(nodes_df4['y_nm']) == 50)]['ID'].to_numpy()
nodes_map4 = nodes_df4.set_index('ID')[['x', 'y']].to_dict('index')
nodes_df6 = pd.read_csv(r'C:\Users\kacpe\Desktop\Kody Warte Uwagi\MOFiT2\Projekt\wezly_N_6_L100.dat', sep='\s+', header=None, names=['ID', 'x_nm', 'y_nm'])
nodes_df6['x'] = nodes_df6['x_nm'] / nm_to_bohr
nodes_df6['y'] = nodes_df6['y_nm'] / nm_to_bohr
idx_border6 = nodes_df6[(np.abs(nodes_df6['x_nm']) == 50) | (np.abs(nodes_df6['y_nm']) == 50)]['ID'].to_numpy()
nodes_map6 = nodes_df6.set_index('ID')[['x', 'y']].to_dict('index')


# 2. Wczytanie mapowania lokalnego na globalne
# Plik: nlg_N_2_L_100.dat
nlg_df2 = pd.read_csv(r'C:\Users\kacpe\Desktop\Kody Warte Uwagi\MOFiT2\Projekt\nlg_N_2_L100.dat', sep='\s+', header=None, names=['ElementID', 'LocalID', 'GlobalID'])
nlg_map2 = nlg_df2.set_index(['ElementID', 'LocalID'])['GlobalID'].to_dict()
nlg_df4 = pd.read_csv(r'C:\Users\kacpe\Desktop\Kody Warte Uwagi\MOFiT2\Projekt\nlg_N_4_L100.dat', sep='\s+', header=None, names=['ElementID', 'LocalID', 'GlobalID'])
nlg_map4 = nlg_df4.set_index(['ElementID', 'LocalID'])['GlobalID'].to_dict()
nlg_df6 = pd.read_csv(r'C:\Users\kacpe\Desktop\Kody Warte Uwagi\MOFiT2\Projekt\nlg_N_6_L100.dat', sep='\s+', header=None, names=['ElementID', 'LocalID', 'GlobalID'])
nlg_map6 = nlg_df6.set_index(['ElementID', 'LocalID'])['GlobalID'].to_dict()

# Kwadratury Gausa w = [w1, w2, w3, w4]
w1 = w2 = (18.0 + np.sqrt(30.0)) / 36.0
w3 = w4 = (18.0 - np.sqrt(30.0)) / 36.0
w = np.array([w1, w2, w3, w4])
p1 = np.sqrt(3.0 / 7.0 - 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
p2 = np.sqrt(3.0 / 7.0 + 2.0 / 7.0 * np.sqrt(6.0 / 5.0))
p = np.array([-p1, p1, p2, -p2])

@njit
def q(xi, i):

    if i == 1:
        return 0.5 * xi * (xi - 1.0)
    elif i == 2:
        return 1.0 - xi**2
    elif i == 3:
        return 0.5 * xi * (xi + 1.0)
    return 0.0

@njit
def h(xi1, xi2, i):

    if i == 1:
        return q(xi1, 1) * q(xi2, 1)
    elif i == 2:
        return q(xi1, 3) * q(xi2, 1)
    elif i == 3:
        return q(xi1, 1) * q(xi2, 3)
    elif i == 4:
        return q(xi1, 3) * q(xi2, 3)
    elif i == 5:
        return q(xi1, 2) * q(xi2, 1)
    elif i == 6:
        return q(xi1, 3) * q(xi2, 2)
    elif i == 7:
        return q(xi1, 1) * q(xi2, 2)
    elif i == 8:
        return q(xi1, 2) * q(xi2, 3)
    elif i == 9:
        return q(xi1, 2) * q(xi2, 2)
    return 0.0

@njit
def calculate_Smatrix(i, j, N, D):
    S_local = np.zeros((9, 9))
    a = D/nm_to_bohr / (2*N)
    C = a**2 / 4.0
    sum_val = 0.0
    for l in range(4):   # xi1 -> p[l]
        for n in range(4): # xi2 -> p[n]
            sum_val += w[l] * w[n] * h(p[l], p[n], j) * h(p[l], p[n], i)
    S_local[j - 1, i - 1] = C * sum_val
    return S_local

# ======= Energia kinetyczna - lokalna =======

@njit
def dq(xi, i):

    if i == 1:
        return xi - 0.5
    elif i == 2:
        return -2.0 * xi
    elif i == 3:
        return xi + 0.5
    return 0.0

@njit
def dh_dxi1(xi1, xi2, i):

    if i == 1:
        return dq(xi1, 1) * q(xi2, 1)
    elif i == 2:
        return dq(xi1, 3) * q(xi2, 1)
    elif i == 3:
        return dq(xi1, 1) * q(xi2, 3)
    elif i == 4:
        return dq(xi1, 3) * q(xi2, 3)
    elif i == 5:
        return dq(xi1, 2) * q(xi2, 1)
    elif i == 6:
        return dq(xi1, 3) * q(xi2, 2)
    elif i == 7:
        return dq(xi1, 1) * q(xi2, 2)
    elif i == 8:
        return dq(xi1, 2) * q(xi2, 3)
    elif i == 9:
        return dq(xi1, 2) * q(xi2, 2)
    return 0.0

@njit
def dh_dxi2(xi1, xi2, i):

    if i == 1:
        return q(xi1, 1) * dq(xi2, 1)
    elif i == 2:
        return q(xi1, 3) * dq(xi2, 1)
    elif i == 3:
        return q(xi1, 1) * dq(xi2, 3)
    elif i == 4:
        return q(xi1, 3) * dq(xi2, 3)
    elif i == 5:
        return q(xi1, 2) * dq(xi2, 1)
    elif i == 6:
        return q(xi1, 3) * dq(xi2, 2)
    elif i == 7:
        return q(xi1, 1) * dq(xi2, 2)
    elif i == 8:
        return q(xi1, 2) * dq(xi2, 3)
    elif i == 9:
        return q(xi1, 2) * dq(xi2, 2)
    return 0.0

@njit
def calculate_Tmatrix(i, j):
    T_local = np.zeros((9, 9))
    C = hbar**2 / 2 / m_eff
    sum_val = 0.0
    for l in range(4):   # xi1 -> p[l]
        for n in range(4): # xi2 -> p[n]
            ilo = (dh_dxi1(p[l], p[n], j) * dh_dxi1(p[l], p[n], i) + dh_dxi2(p[l], p[n], j) * dh_dxi2(p[l], p[n], i))
            sum_val += w[l] * w[n] * ilo
    T_local[j - 1, i - 1] = C * sum_val
    return T_local

# ====== Energia potencjalna - k = 11 =======
@njit
def g(xi1, xi2, i):

    if i == 1:
        return (0.5 * (1.0 - xi1)) * (0.5 * (1.0 - xi2))
    elif i == 2:
        return (0.5 * (1.0 + xi1)) * (0.5 * (1.0 - xi2))
    elif i == 3:
        return (0.5 * (1.0 - xi1)) * (0.5 * (1.0 + xi2))
    elif i == 4:
        return (0.5 * (1.0 + xi1)) * (0.5 * (1.0 + xi2))
    return 0.0

def calculate_xy(l, n, k, D, nodes_coords, nlg_map):
    x = 0.0
    y = 0.0
    for corner_idx in range(4):
        global_id = nlg_map[(k, corner_idx+1)]

        x_nlg = nodes_coords[global_id]['x'] * D/L
        y_nlg = nodes_coords[global_id]['y'] * D/L
        x += x_nlg * g(p[l], p[n], corner_idx + 1)
        y += y_nlg * g(p[l], p[n], corner_idx + 1)
    return x, y

def calculate_Vmatrix(i, j, k, D, N, nodes_coords, nlg_map):
    V_local = np.zeros((9, 9))
    a = D/nm_to_bohr / (2*N)
    C = a**2/4 * m_eff*omega**2/2
    sum_val = 0.0
    for l in range(4):   # xi1 -> p[l]
        for n in range(4): # xi2 -> p[n]
            x, y = calculate_xy(l=l, n=n, k=k, D=D, nodes_coords=nodes_coords, nlg_map=nlg_map)
            xy = x**2 + y**2
            sum_val += w[l] * w[n] * xy * h(p[l], p[n], j) * h(p[l], p[n], i)
    V_local[j - 1, i - 1] = C * sum_val
    return V_local

# ====== Składamy macierze globalne =======

def calculate_global_S_H(N, D, nodes_coords, nlg_map, idx_border):
    N_global = (4*N+1)**2
    S_global = np.zeros((N_global, N_global), dtype=np.float64)
    H_global = np.zeros((N_global, N_global), dtype=np.float64)
    element_ids = list(set(k for k,_ in nlg_map.keys()))


    for k in element_ids:
        for j in range(1, 10):
            for i in range(1, 10):
                S = calculate_Smatrix(i=i, j=j, N=N, D=D)[j-1, i-1]
                T = calculate_Tmatrix(i=i, j=j)[j-1, i-1]
                V = calculate_Vmatrix(i=i, j=j, k=k, D=D, N=N, nodes_coords=nodes_coords, nlg_map=nlg_map)[j-1, i-1]
                H = T + V
                row = nlg_map[(k, j)] - 1
                col = nlg_map[(k, i)] - 1
                S_global[row, col] += S
                H_global[row, col] += H

    for idx in idx_border:
        S_global[idx-1, :] = 0
        S_global[:, idx-1] = 0
        H_global[idx-1, :] = 0
        H_global[:, idx-1] = 0
        S_global[idx-1, idx-1] = 1
        H_global[idx-1, idx-1] = -1410
    return S_global, H_global

def solve_generalized_eig(H, S, num_states=12, positive_only=True):

    E_all, C_all = scipy.linalg.eigh(H, S)
    idx = np.argsort(E_all)
    E_all = E_all[idx]
    C_all = C_all[:, idx]
    if positive_only:
        pos_mask = E_all > 0
        E = E_all[pos_mask]
        C = C_all[:, pos_mask]
    else:
        E = E_all
        C = C_all
    if num_states is not None:
        E = E[:num_states]
        C = C[:, :len(E)]
    return E, C

# Najpierw wyznaczono macierze lokalne przekrywania S, energii kinetycznej T oraz k=16 energii potencjalnej V dla N=2 oraz L=100 nm.
# Otrzymane wyniki porównano z wynikami przekazanymi przez prowadzącego - otrzymano zgodność do 2 miejsc po przecinku.
# Dla macierzy V wprowadzono indeksowanie po k=16 elementach i także porównano otrzymany wynik, wcześniej wczytując potrzebne dane z plików nlg_N_2_L100.dat
# oraz wezly_N_2_L100.dat.
# Po udanym uzyskaniu powyższych wyników złożono macierze globalne. Po tym kroku także uzyskano zgodny z wynikiem prowadzącego rezultat.
# Dzięki uzyskanym macierzom S, oraz H=T+V rozwiązane zostało równanie własne:
#   Hc = ESc,
# używając metody scipy.linalg.eigh, uzyskując tym samym wartości własne (energie) oraz funkcje własne c spełniające zadany układ.
# Procedure obliczeń E oraz c powtórzono dla różnych wartości dla których to pliki pomocnicze do mapowania zostały podane przez prowadzącego N={2, 4, 6}.
# Po tym, dla każdego z N znaleziono minimalne L (D), dla którego uzyskujemy minimalną energię. Spośród trzech wyników (dla każdego N) najmniejszą
# energię uzyskano dla N=6, L=72. Następnie do obliczeń przyjęto takie właśnie wartości jako najbardziej optymalne.
# Dla uzyskanych wyżej wartości narysowano pierwsze 6 (o najniższej energii) funkcji falowych. W tym celu zagęszczono siatkę mapując dla każdego
# elementu punkty xi1, xi2 zgodnie z instrukcją. Wyniki przedstawiono na Rys. \ref{}.
# Widzimy pierwszy stan podstawowy o najniższej energii oraz kolejne stany wbudzone i ich zmienną funkcję przestrzenną (kształt).


N2 = 2
N4 = 4
N6 = 6

S2_glob, H2_glob = calculate_global_S_H(N=N2, D=100, nodes_coords=nodes_map2, nlg_map=nlg_map2, idx_border=idx_border2)
S4_glob, H4_glob = calculate_global_S_H(N=N4, D=100, nodes_coords=nodes_map4, nlg_map=nlg_map4, idx_border=idx_border4)
S6_glob, H6_glob = calculate_global_S_H(N=N6, D=100, nodes_coords=nodes_map6, nlg_map=nlg_map6, idx_border=idx_border6)

for i in range(81):
    for j in range(81):
        print(i+1,j+1, S2_glob[i,j], H2_glob)

E2, C2 = solve_generalized_eig(H=H2_glob, S=S2_glob, num_states=10, positive_only=True)
E4, C4 = solve_generalized_eig(H=H4_glob, S=S4_glob, num_states=10, positive_only=True)
E6, C6 = solve_generalized_eig(H=H6_glob, S=S6_glob, num_states=10, positive_only=True)


E2_meV = E2 * 27211.6
print("Energia dla N=2, L=100: \n", E2_meV)
E4_meV = E4 * 27211.6
print("Energia dla N=4, L=100: \n", E4_meV)
E6_meV = E6 * 27211.6
print("Energia dla N=6, L=80: \n", E6_meV)

def find_best_L(N, L_values, nodes_coords, nlg_map, idx_border):
    ground_energies1 = []
    ground_energies = []
    for Di in L_values:
        S, H = calculate_global_S_H(N=N, D=Di, nodes_coords=nodes_coords, nlg_map=nlg_map, idx_border=idx_border)
        E, C = solve_generalized_eig(H=H, S=S, num_states=10, positive_only=True)
        ground_energies1.append(E[0])
        ground_energies.append(np.sum(E[:6]))

    idx_minE = np.argmin(ground_energies)
    idx_minE1 = np.argmin(ground_energies1)
    L_best = L_values[idx_minE]
    E_best = ground_energies[idx_minE]
    E_best1 = ground_energies1[idx_minE1]


    print(f"Dla N = {N}, optymalne L = {L_best}, suma najmniejszych 6 energii = {E_best}, energia podstawowa = {E_best1}")

    return ground_energies, L_best, E_best


def calculate_psi(xi1, xi2, k, state_vec, nlg_map):
    psi = 0.0
    for i in range(1, 10):
        global_id = nlg_map[(k, i)]
        psi_nlg = state_vec[global_id - 1]
        psi += psi_nlg * h(xi1, xi2, i)
    return psi

def map_element(xi1, xi2, k, D, nodes_coords, nlg_map):
    x = 0.0
    y = 0.0
    for corner_idx in range(4): 
        global_id = nlg_map[(k, corner_idx+1)] 
        x_nlg = nodes_coords[global_id]['x'] * D / L
        y_nlg = nodes_coords[global_id]['y'] * D / L
        x += x_nlg * g(xi1, xi2, corner_idx + 1)
        y += y_nlg * g(xi1, xi2, corner_idx + 1)
    return x, y

def generate_dense_psi_data(N, D, state_vec, nodes_coords, nlg_map, resolution=20):
    
    xi = np.linspace(-1, 1, resolution)
    Xi1, Xi2 = np.meshgrid(xi, xi)
    xi1_flat = Xi1.flatten()
    xi2_flat = Xi2.flatten()
    X_dense, Y_dense, Psi_dense = [], [], []

    element_ids = sorted(list(set(k for k, _ in nlg_map.keys())))
    for k in element_ids:
        for xi1, xi2 in zip(xi1_flat, xi2_flat):
            x, y = map_element(xi1, xi2, k, D, nodes_coords, nlg_map)
            psi = calculate_psi(xi1, xi2, k, state_vec, nlg_map)
            
            X_dense.append(x)
            Y_dense.append(y)
            Psi_dense.append(psi)

    X_dense = np.array(X_dense) * nm_to_bohr
    Y_dense = np.array(Y_dense) * nm_to_bohr
    Psi_dense = np.array(Psi_dense)
    
    return X_dense, Y_dense, Psi_dense

def draw_n_states(Cbest, n_states=int, resolution=int):
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    axes = axes.flatten()
    print("Najniższe energie:", Ebest[:n_states] * 27211.6)

    for n in range(n_states):
        C_n = Cbest[:, n]
        
        X_dense, Y_dense, Psi_dense = generate_dense_psi_data(N=N_best, D=L_best, state_vec=C_n, nodes_coords=nodes_map6, nlg_map=nlg_map6, resolution=resolution) # resolution=10 to 10x10 punktów na element
        contour = axes[n].tricontourf(X_dense, Y_dense, Psi_dense, levels=50, cmap='RdBu_r') 
        axes[n].set_title(f"Stan n={n+1}, E={Ebest[n] * 27211.6:.2f} meV")
        axes[n].set_xlabel("x [nm]")
        axes[n].set_ylabel("y [nm]")
        axes[n].axis("equal")
        
        fig.colorbar(contour, ax=axes[n], label=f"$\\Psi_{n+1}$")

    plt.tight_layout()
    plt.savefig("n_states.png")
    plt.show()

# ---------- Funkcje do macierzy operatora położenia X (lokalne + składanie) ----------

# operator położenia X to po prostu x -> wzór taki sam jak dla potencjału, ale zamiast V(x,y), jest x
def calculate_Xmatrix(i, j, k, D, N, nodes_coords, nlg_map):
    X_local = np.zeros((9, 9))
    a = D/nm_to_bohr / (2*N)
    C = a**2/4
    sum_val = 0.0
    for l in range(4):
        for n in range(4):
            x, y = calculate_xy(l=l, n=n, k=k, D=D, nodes_coords=nodes_coords, nlg_map=nlg_map)
            sum_val += w[l] * w[n] * x * h(p[l], p[n], j) * h(p[l], p[n], i)
    X_local[j - 1, i - 1] = C * sum_val
    return X_local

def calculate_global_X(N, D, nodes_coords, nlg_map, idx_border):
    N_global = (4*N+1)**2
    X_global = np.zeros((N_global, N_global), dtype=np.float64)
    element_ids = list(set(k for k, _ in nlg_map.keys()))


    for k in element_ids:
        for j in range(1, 10):
            for i in range(1, 10):
                X = calculate_Xmatrix(i=i, j=j, k=k, D=D, N=N, nodes_coords=nodes_coords, nlg_map=nlg_map)[j-1, i-1]
                row = nlg_map[(k, j)] - 1
                col = nlg_map[(k, i)] - 1
                X_global[row, col] += X

    for idx in idx_border:
        X_global[idx-1, :] = 0
        X_global[:, idx-1] = 0
        X_global[idx-1, idx-1] = 1

    return X_global

# ---------- Ewolucja w czasie ----------

def time_evolution_CN(S, H, d0, dt, t_final):

    A = (S.astype(np.complex128) + 1j * dt / (2*hbar) * H.astype(np.complex128))
    B = (S.astype(np.complex128) - 1j * dt / (2*hbar) * H.astype(np.complex128))

    times = np.arange(0, t_final*4 + 1e-12, dt)
    Ds = np.zeros((len(times), len(d0)), dtype=np.complex128)
    Ds[0, :] = d0.astype(np.complex128)

    for t in range(len(times)-1):
        rhs = B.dot(Ds[t, :])
        Ds[t+1, :] = scipy.linalg.solve(A, rhs)

    return times, Ds


def run_time_evolution_and_plots(S_glob, H_glob, E_vals, C_vecs,
                                  nodes_coords, nlg_map, idx_border,
                                  N, D,
                                  dt=100.0,
                                  snapshots=4, resolution=20,
                                  plot_psi_snapshots=True):
    
    DeltaE = E_vals[1] - E_vals[0]
    T = 2.0 * np.pi / DeltaE
    print(f"DeltaE = {DeltaE:.4e} [jedn. atomowe], okres T = {T:.4e} [jedn. atomowe czasu]")

    X_glob = calculate_global_X(N=N, D=D, nodes_coords=nodes_coords, nlg_map=nlg_map, idx_border=idx_border)

    # stan początkowy: kombinacja pierwszego i drugiego stanu
    d0 = C_vecs[:, 0] + C_vecs[:, 1]

    times, Ds = time_evolution_CN(S_glob, H_glob, d0, dt, T)

    # wykres x(t)
    x_t = np.zeros(len(times), dtype=np.complex128)
    for t in range(len(times)):
        d = Ds[t, :]
        x_t[t] = np.vdot(d, X_glob.dot(d))

    # klatki
    snapshot_times = [k * T / snapshots for k in range(snapshots)]
    snapshot_indices = [int(np.round(ti / dt1)) for ti in snapshot_times]
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    for ax, idx_snap, t_snap in zip(axes, snapshot_indices, snapshot_times):
        d_snap = Ds[idx_snap, :]
        X_dense, Y_dense, Psi_dense = generate_dense_psi_data(
            N=N, D=D, state_vec=d_snap, nodes_coords=nodes_map6, nlg_map=nlg_map6, resolution=resolution
        )
        dens = np.abs(Psi_dense)**2
        contour = ax.tricontourf(X_dense * nm_to_bohr, Y_dense * nm_to_bohr, dens, levels=60)
        ax.set_title(f"t = {t_snap:.3e} a.u, dt = {dt1}")
        ax.set_xlabel('x [nm]')
        ax.set_ylabel('y [nm]')
        ax.axis('equal')
        fig.colorbar(contour, ax=ax, label=r'$|\Psi|^2$')

    plt.tight_layout()
    plt.show()
    return times, x_t, X_glob, Ds


# Minimalizacja energii odbywa się poprzez wzięcie pierwszych 6 najniższych stanów energetycznych i ich zsumowaniu. Następnie otrzymana
# suma jest sprawdzana pod kątem "minimalnej". Jest tak dlatego, że same stany podstawowe nie wystarczą, większe różnice w energiach
# zaczynają pojawiać się dopiero dla wyższych stanów. Wzięto 6 najniższych do obliczeń, gdyż dla tych chcemy zobaczyć póżniej rysunki.
# Najlepsze N zostało wybrane na podstawie różnic w energii podstawowej, w stosunku do N=4 różnica ta już jest niewielka w porównaniu między N=2, a N=4.
# Stąd wzięto najwyższe możliwe N=6 dla którego zostały przygotowane pliki z mapowaniem.
# Ostatecznie więc otrzymano najlepsze N=6, a dla niego najlepsze L=80 na podstawie użytego algorytmu sprawdzania sum pierwszych 6 energii.

L_values = np.linspace(40, 120, 41)
ground_energies2, L_best2, E_best2 = find_best_L(N=N2, L_values=L_values, nodes_coords=nodes_map2, nlg_map=nlg_map2, idx_border=idx_border2)
ground_energies4, L_best4, E_best4 = find_best_L(N=N4, L_values=L_values, nodes_coords=nodes_map4, nlg_map=nlg_map4, idx_border=idx_border4)
ground_energies6, L_best6, E_best6 = find_best_L(N=N6, L_values=L_values, nodes_coords=nodes_map6, nlg_map=nlg_map6, idx_border=idx_border6)

plt.figure(figsize=(8,5))
plt.plot(L_values, np.array(ground_energies2) * 27211.6, '-', label="N=2")
plt.plot(L_values, np.array(ground_energies4) * 27211.6, '-', label="N=4")
plt.plot(L_values, np.array(ground_energies6) * 27211.6, '-', label="N=6")
plt.xlabel("L [nm]")
plt.ylabel("Suma pierwszych 6 energii [meV]")
plt.legend()
plt.grid(True)
plt.savefig("L_vs_E.png")
plt.show()

N_best = N6
L_best = L_best6

Sbest_glob, Hbest_glob = calculate_global_S_H(N=N_best, D=L_best, nodes_coords=nodes_map6, nlg_map=nlg_map6, idx_border=idx_border6)
Ebest, Cbest = solve_generalized_eig(H=Hbest_glob, S=Sbest_glob, num_states=10, positive_only=True)
draw_n_states(Cbest=Cbest, n_states=6, resolution=10)

dt1 = 100
dt2 = 200
dt3 = 300
dt4 = 500
snapshots=4
resolution=18
times1, x_t1, X_glob1, Ds1 = run_time_evolution_and_plots(Sbest_glob, Hbest_glob, Ebest, Cbest, nodes_map6, nlg_map6, idx_border6, N_best, L_best, dt=dt1)
times2, x_t2, X_glob2, Ds2 = run_time_evolution_and_plots(Sbest_glob, Hbest_glob, Ebest, Cbest, nodes_map6, nlg_map6, idx_border6, N_best, L_best, dt=dt2)
times3, x_t3, X_glob3, Ds3 = run_time_evolution_and_plots(Sbest_glob, Hbest_glob, Ebest, Cbest, nodes_map6, nlg_map6, idx_border6, N_best, L_best, dt=dt3)
times4, x_t4, X_glob4, Ds4 = run_time_evolution_and_plots(Sbest_glob, Hbest_glob, Ebest, Cbest, nodes_map6, nlg_map6, idx_border6, N_best, L_best, dt=dt4)

# x(t)
plt.figure(figsize=(8,4))
plt.plot(times1, x_t1.real * nm_to_bohr, label="dt=100")
plt.plot(times2, x_t2.real * nm_to_bohr, label="dt=200")
plt.plot(times3, x_t3.real * nm_to_bohr, label="dt=300")
plt.plot(times4, x_t4.real * nm_to_bohr, label="dt=500")
plt.xlabel('t [a.u.]')
plt.ylabel('x(t) [nm]')
plt.title(r'$x(t) = d^\dagger X d$')
plt.grid(True)
plt.legend()
plt.savefig("x_t.png")
plt.show()




# C_1 = Cbest[:,0]
# N_global = len(C_1)
# x = np.zeros(N_global)
# y = np.zeros(N_global)
# for idx in range(N_global):
#     global_id = idx + 1
#     x[idx] = nodes_map6[global_id]['x'] * D/L
#     y[idx] = nodes_map6[global_id]['y'] * D/L
# # rysowanie
# plt.figure(figsize=(6,5))
# plt.tricontourf(x * nm_to_bohr, y * nm_to_bohr, C_1, levels=50)
# plt.colorbar(label="ψ")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Mapa 2D funkcji własnej")
# plt.axis("equal")
# plt.show()











