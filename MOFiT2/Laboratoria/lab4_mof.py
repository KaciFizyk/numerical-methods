import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ----- Parametry -----
h = 1.0545718e-34
m0 = 9.10938356e-31
m_eff = 0.067 * m0
eV = 1.602176634e-19 

x_min, x_max = -100*1e-9, 100*1e-9
Nx = 201
x = np.linspace(x_min, x_max, Nx)
dx = x[1] - x[0]

hbar_omega = 5 * eV * 1e-3
omega = hbar_omega / h
V = 0.5 * m_eff * (omega**2) * (x**2)
T = 2 * np.pi / omega
dt = 1.0
dt_SI = dt * 2.42e-17
t = np.arange(0, 10*T, dt_SI)
Nt = len(t)

# ----- Normalizacja -----
@njit
def normalize(psi, dx):
    C = np.sum(np.abs(psi)**2) * dx
    return psi / np.sqrt(C)
# ----- Hamiltonian na Psi -----
@njit
def hamiltonian(psi_t, V, dx):
    N = len(psi_t)
    Hpsi = np.zeros(N, dtype=np.complex128)
    for i in range(1, N-1):
        Hpsi[i] = - (h**2 / (2*m_eff)) * ((psi_t[i+1] + psi_t[i-1] - 2 * psi_t[i]) / dx**2) + V[i] * psi_t[i]
    return Hpsi
# ----- Krok czasowy - metoda Askara -----
@njit
def evolve_askar(psi, V, dx, dt):
    x_steps, t_steps = psi.shape
    for t_step in range(1, t_steps - 1):
        Hpsi = hamiltonian(psi[:, t_step], V, dx)
        psi[:, t_step+1] = psi[:, t_step-1] - (2j * dt / h) * Hpsi
        psi[ 0, t_step+1] = 0
        psi[-1, t_step+1] = 0
    return psi



# ----- ZADANIE 3 ----- Funkcja falowa dla x0 = 30 nm -----

x0 = 30.0*1e-9
psi0 = np.exp(-m_eff * omega * (x - x0)**2 / (2*h))
psi0_norm = normalize(psi0, dx)

psi = np.zeros((Nx, Nt), dtype=np.complex128)
psi[:, 0] = psi0_norm
psi[:, 1] = psi0_norm * np.exp(-1j * omega * dt_SI / 2)


psi = evolve_askar(psi, V, dx, dt_SI)

# Wykres Psi dla t = 0 s
plt.figure(figsize=(8,6))
plt.plot(x / 1e-9, np.abs(psi[:, 0]*np.sqrt(1e-9))**2, label=f"t = {0 * dt_SI * 1e12:.2f} ps")

plt.xlabel("x [nm]")
plt.ylabel(r'$|\Psi(x,t)|^2$')
plt.title("Ewolucja funkcji falowej w oscylatorze harmonicznym, t=0 s")
plt.legend()
plt.tight_layout()
plt.show()

# Wykres Psi dla różnych t - mało widać bo bardzo dużo chwil t
plt.figure(figsize=(8,6))
for t_step in range(0, Nt, 10000):
    plt.plot(x / 1e-9, np.abs(psi[:, t_step]*np.sqrt(1e-9))**2, label=f"t = {t_step * dt_SI * 1e12:.2f} ps")
plt.xlabel("x [nm]")
plt.ylabel(r'$|\Psi(x,t)|^2$')
plt.title("Ewolucja funkcji falowej w oscylatorze harmonicznym")
# plt.legend()
plt.tight_layout()
plt.show()

# Animacja tego co wyżej - lepiej widać ewolucję
from matplotlib.animation import FuncAnimation

scale = np.sqrt(1e-9)
fig, ax = plt.subplots(figsize=(8,6))
line, = ax.plot([], [], lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(np.min(x)/1e-9, np.max(x)/1e-9)
ax.set_ylim(0, np.max(np.abs(psi*scale)**2)*1.1)  # lekki margines
ax.set_xlabel("x [nm]")
ax.set_ylabel(r'$|\Psi(x,t)|^2$')
ax.set_title("Ewolucja funkcji falowej w oscylatorze harmonicznym")

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    # frame to indeks kroku czasowego
    y = np.abs(psi[:, frame] * scale)**2
    line.set_data(x/1e-9, y)
    time_text.set_text(f"t = {frame*dt_SI*1e12:.2f} ps")
    return line, time_text

frames_to_show = np.arange(0, Nt, 500)
ani = FuncAnimation(fig, update, frames=frames_to_show, init_func=init,
                    blit=True, interval=50)  # interval w ms
plt.show()
# ani.save("ewolucja1.gif", writer='pillow', fps=30)


# Wykres ewolucji czasowej 3d, widać lepiej jak się zmienia w czasie położenie
plt.figure(figsize=(8,5))
plt.imshow(np.abs(psi*np.sqrt(1e-9))**2, 
           aspect='auto',
           origin='lower',
           extent=[t[0]*1e12, t[-1]*1e12, x[0]*1e9, x[-1]*1e9])  # X = czas [ps], Y = x [nm]
plt.xlabel("t [ps]")
plt.ylabel("x [nm]")
plt.title(r'$|\Psi(x,t)|^2$ w funkcji czasu')
plt.colorbar(label=r'$|\Psi(x,t)|^2$')
plt.show()

# ----- ZADANIE 4 ----- Wartości oczekiwane -----
@njit
def expected_value(psi):
    x_expectation = np.zeros(Nt)

    for t_step in range(Nt):
        x_expectation[t_step] = np.sum(x * np.abs(psi[:, t_step])**2) * dx
    return x_expectation

@njit
def expected_value_classic():
    x_classical = x0 * np.cos(omega * t)
    return x_classical

plt.figure(figsize=(8,6))
plt.plot(t*1e12, expected_value(psi)/1e-9, label="Wartość oczekiwana <x(t)>")
plt.plot(t*1e12, expected_value_classic()/1e-9, '--', label="Klasyczne x(t)")
plt.xlabel("t [ps]")
plt.ylabel("Położenie x [nm]")
plt.title("Porównanie wartości oczekiwanej z ruchem klasycznym")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# ----- ZADANIE 5 ----- Funkcja falowa dla x0 = 0 nm -----
# Gęstość prawdopodobieństwa nie zmienia się w czasie, stan podstawowy jest stacjonarny, nie "ulega" potencjałowi oscylatora.
# Wartość oczekiwana położenia nie zmienia się w czasie. Jeśli wrzucić pakiet do środka potencjału nie ma powodu by się ruszał
# - jak klasycznie - równowaga stała.
x0 = 0
psi0 = np.exp(-m_eff * omega * (x - x0)**2 / (2*h))

psi0_norm = normalize(psi0, dx)

psi = np.zeros((Nx, Nt), dtype=np.complex128)
psi[:, 0] = psi0_norm
psi[:, 1] = psi0_norm * np.exp(-1j * omega * dt_SI / 2)


psi = evolve_askar(psi, V, dx, dt_SI)

# Wykres Psi dla t = 0 s
plt.figure(figsize=(8,6))
plt.plot(x / 1e-9, np.abs(psi[:, 0]*np.sqrt(1e-9))**2, label=f"t = {0 * dt_SI * 1e12:.2f} ps")

plt.xlabel("x [nm]")
plt.ylabel(r'$|\Psi(x,t)|^2$')
plt.title("Ewolucja funkcji falowej w oscylatorze harmonicznym, t=0 s, x0=0")
plt.legend()
plt.tight_layout()
plt.show()

# Wykres Psi dla różnych t - mało widać bo bardzo dużo chwil t
plt.figure(figsize=(8,6))
for t_step in range(0, Nt, 10000):
    plt.plot(x / 1e-9, np.abs(psi[:, t_step]*np.sqrt(1e-9))**2, label=f"t = {t_step * dt_SI * 1e12:.2f} ps")
plt.xlabel("x [nm]")
plt.ylabel(r'$|\Psi(x,t)|^2$')
plt.title("Ewolucja funkcji falowej w oscylatorze harmonicznym, x0=0")
# plt.legend()
plt.tight_layout()
plt.show()

# Animacja tego co wyżej - lepiej widać ewolucję
scale = np.sqrt(1e-9)
fig, ax = plt.subplots(figsize=(8,6))
line, = ax.plot([], [], lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(np.min(x)/1e-9, np.max(x)/1e-9)
ax.set_ylim(0, np.max(np.abs(psi*scale)**2)*1.1)  # lekki margines
ax.set_xlabel("x [nm]")
ax.set_ylabel(r'$|\Psi(x,t)|^2$')
ax.set_title("Ewolucja funkcji falowej w oscylatorze harmonicznym, x0=0")

frames_to_show = np.arange(0, Nt, 500)
ani = FuncAnimation(fig, update, frames=frames_to_show, init_func=init,
                    blit=True, interval=50)  # interval w ms
plt.show()
# ani.save("ewolucja2_full.gif", writer='pillow', fps=30)

plt.figure(figsize=(8,6))
plt.plot(t*1e12, expected_value(psi)/1e-9, label="Wartość oczekiwana <x(t)>")
plt.plot(t*1e12, np.zeros_like(t), '--', label="Klasyczne x(t)")
plt.xlabel("t [ps]")
plt.ylabel("Położenie x [nm]")
plt.title("Porównanie wartości oczekiwanej z ruchem klasycznym")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Wykres ewolucji czasowej 3d, widać lepiej jak się zmienia w czasie położenie
plt.figure(figsize=(8,5))
plt.imshow(np.abs(psi*np.sqrt(1e-9))**2, 
           aspect='auto',
           origin='lower',
           extent=[t[0]*1e12, t[-1]*1e12, x[0]*1e9, x[-1]*1e9])  # X = czas [ps], Y = x [nm]
plt.xlabel("t [ps]")
plt.ylabel("x [nm]")
plt.title(r'$|\Psi(x,t)|^2$ w funkcji czasu')
plt.colorbar(label=r'$|\Psi(x,t)|^2$')
plt.show()



# ----- ZADANIE 6 ----- Funkcja falowa dla x0 = 0 nm, brak potencjału -----
# Funkcja falowa jest skupiona w centrum studni, a następnie rozchodzi się po całej studni.
# Warunki brzegowe powodują, że fale odbijają się od końców studni. 
# 
# Pakiet falowy może być przedstawiony jako superpozycja stanów własnych w studni (sinusy), a każdy z nich ewouuluje w czasie 
# co powoduje rozmycie się pakietu. Superpozycja wielu stanów -> interferencja między nimi.
#
# Po pewnym czasie pakiet może częściowo lub całkowicie wrócić do stanu początkowego.

psi0 = np.exp(-m_eff * omega * (x - x0)**2 / (2*h))
V = np.zeros_like(x)

psi0_norm = normalize(psi0, dx)

psi = np.zeros((Nx, Nt), dtype=np.complex128)
psi[:, 0] = psi0_norm
psi[:, 1] = psi0_norm * np.exp(-1j * omega * dt_SI / 2)


psi = evolve_askar(psi, V, dx, dt_SI)

# Wykres Psi dla t = 0 s
plt.figure(figsize=(8,6))
plt.plot(x / 1e-9, np.abs(psi[:, 0]*np.sqrt(1e-9))**2, label=f"t = {0 * dt_SI * 1e12:.2f} ps")

plt.xlabel("x [nm]")
plt.ylabel(r'$|\Psi(x,t)|^2$')
plt.title("Ewolucja funkcji falowej w oscylatorze harmonicznym, t=0 s, x0=0, V=0")
plt.legend()
plt.tight_layout()
plt.show()

# Wykres Psi dla różnych t - mało widać bo bardzo dużo chwil t
plt.figure(figsize=(8,6))
for t_step in range(0, Nt, 10000):
    plt.plot(x / 1e-9, np.abs(psi[:, t_step]*np.sqrt(1e-9))**2, label=f"t = {t_step * dt_SI * 1e12:.2f} ps")
plt.xlabel("x [nm]")
plt.ylabel(r'$|\Psi(x,t)|^2$')
plt.title("Ewolucja funkcji falowej w oscylatorze harmonicznym, x0=0, V=0")
# plt.legend()
plt.tight_layout()
plt.show()

# Animacja tego co wyżej - lepiej widać ewolucję
scale = np.sqrt(1e-9)
fig, ax = plt.subplots(figsize=(8,6))
line, = ax.plot([], [], lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_xlim(np.min(x)/1e-9, np.max(x)/1e-9)
ax.set_ylim(0, np.max(np.abs(psi*scale)**2)*1.1)  # lekki margines
ax.set_xlabel("x [nm]")
ax.set_ylabel(r'$|\Psi(x,t)|^2$')
ax.set_title("Ewolucja funkcji falowej w oscylatorze harmonicznym, x0=0, V=0")

frames_to_show = np.arange(0, Nt, 500)
ani = FuncAnimation(fig, update, frames=frames_to_show, init_func=init,
                    blit=True, interval=50)  # interval w ms
plt.show()
# ani.save("ewolucja3.gif", writer='pillow', fps=30)

plt.figure(figsize=(8,6))
plt.plot(t*1e12, expected_value(psi)/1e-9, label="Wartość oczekiwana <x(t)>")
# plt.plot(t*1e12, expected_value_classic()/1e-9, '--', label="Klasyczne x(t)")
plt.xlabel("t [ps]")
plt.ylabel("Położenie x [nm]")
plt.title("Porównanie wartości oczekiwanej z ruchem klasycznym")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Wykres ewolucji czasowej 3d, widać lepiej jak się zmienia w czasie położenie
plt.figure(figsize=(8,5))
plt.imshow(np.abs(psi*np.sqrt(1e-9))**2, 
           aspect='auto',
           origin='lower',
           extent=[t[0]*1e12, t[-1]*1e12, x[0]*1e9, x[-1]*1e9])  # X = czas [ps], Y = x [nm]
plt.xlabel("t [ps]")
plt.ylabel("x [nm]")
plt.title(r'$|\Psi(x,t)|^2$ w funkcji czasu')
plt.colorbar(label=r'$|\Psi(x,t)|^2$')
plt.show()