#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
using namespace std;

// Parametry
const double dz = 0.01;
const double Q = -400;
const double mu = 1.0;
const double y1_const = -0.4;
const double y2 = 0.4;

const int x_min = -100, x_max = 150;
const int y_min = -40, y_max = 40;

const int nx = (x_max - x_min) + 1;
const int ny = (y_max - y_min) + 1;

vector<double> x(nx), y(ny);
vector<vector<double>> psi(nx, vector<double>(ny, 0.0));
vector<vector<double>> zeta(nx, vector<double>(ny, 0.0));

double psi_func(double y) {
    return Q / (2.0 * mu) * (pow(y, 3) / 3.0 - 0.5 * pow(y, 2) * (y1_const + y2) + y1_const * y2 * y);
}

double zeta_func(double y) {
    return Q / (2.0 * mu) * (2.0 * y - y1_const - y2);
}

double psi_relax(int i, int j) {
    return (psi[i + 1][j] + psi[i - 1][j] + psi[i][j + 1] + psi[i][j - 1] - zeta[i][j] * dz * dz) / 4.0;
}

double zeta_relax(int i, int j) {
    double dpsidy = (psi[i][j + 1] - psi[i][j - 1]) / (2.0 * dz);
    double dpsidx = (psi[i + 1][j] - psi[i - 1][j]) / (2.0 * dz);
    double dzetadx = (zeta[i + 1][j] - zeta[i - 1][j]) / (2.0 * dz);
    double dzetady = (zeta[i][j + 1] - zeta[i][j - 1]) / (2.0 * dz);

    return (zeta[i + 1][j] + zeta[i - 1][j] + zeta[i][j + 1] + zeta[i][j - 1]) / 4.0
        - (dpsidy * dzetadx - dpsidx * dzetady) * dz * dz / 4.0;
}

int main() {
    // Inicjalizacja siatek
    for (int i = 0; i < nx; ++i) x[i] = (x_min + i) * dz;
    for (int j = 0; j < ny; ++j) y[j] = (y_min + j) * dz;

    const int max_iter = 10000;
    const double tolerance = 1e-10;
    const int check_i = static_cast<int>(0.5 / dz + 100);
    const int check_j = static_cast<int>(0.0 / dz + 40);

    // wymiary zastawki
    const int ik = 5, jk = 10;
    const int i_left = nx - x_max - ik;
    const int i_right = nx - x_max + ik;
    const int j_top = jk - y_min;
    const int j_bot = 0;

    for (int it = 0; it < max_iter; ++it) {
        auto psi_old = psi;
        auto zeta_old = zeta;

        // Warunki brzegowe
        for (int i = 0; i < nx; ++i) {
            psi[i][0] = psi_func(y1_const);
            psi[i][ny - 1] = psi_func(y[ny - 1]);
        }

        for (int j = 0; j < ny; ++j) {
            psi[0][j] = psi_func(y[j]);
            psi[nx - 1][j] = psi_func(y[j]);
        }

        for (int i = i_left + 1; i < i_right - 1; ++i)
            psi[i][j_top] = psi_func(y1_const);

        for (int j = 1; j < j_top; ++j) {
            psi[i_left][j] = psi_func(y1_const);
            psi[i_right][j] = psi_func(y1_const);
        }

        for (int i = 0; i < nx; ++i)
            zeta[i][ny - 1] = 2.0 * (psi[i][ny - 2] - psi[i][ny - 1]) / (dz * dz);

        for (int i = 1; i <= i_left; ++i)
            zeta[i][0] = 2.0 * (psi[i][1] - psi[i][0]) / (dz * dz);
        for (int i = i_right; i < nx - 1; ++i)
            zeta[i][0] = 2.0 * (psi[i][1] - psi[i][0]) / (dz * dz);

        for (int j = 1; j < j_top; ++j) {
            zeta[i_left][j] = 2.0 * (psi[i_left - 1][j] - psi[i_left][j]) / (dz * dz);
            zeta[i_right][j] = 2.0 * (psi[i_right + 1][j] - psi[i_right][j]) / (dz * dz);
        }

        for (int i = i_left + 1; i < i_right - 1; ++i)
            zeta[i][j_top] = 2.0 * (psi[i][j_top + 1] - psi[i][j_top]) / (dz * dz);

        zeta[i_left][j_top] = 0.5 * (
            2.0 * (psi[i_left - 1][j_top] - psi[i_left][j_top]) / (dz * dz) +
            2.0 * (psi[i_left][j_top + 1] - psi[i_left][j_top]) / (dz * dz)
        );

        zeta[i_right][j_top] = 0.5 * (
            2.0 * (psi[i_right + 1][j_top] - psi[i_right][j_top]) / (dz * dz) +
            2.0 * (psi[i_right][j_top + 1] - psi[i_right][j_top]) / (dz * dz)
        );

        if (it % 500 == 0) {
            cout << fixed << setprecision(10)
                 << "iter=" << it << ": psi(50,0)=" << psi[150][40]
                 << ", psi(60,20)=" << psi[160][60] << endl;
        }

        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                psi[i][j] = psi_relax(i, j);

        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                zeta[i][j] = zeta_relax(i, j);

        if (it >= 1000) {
            double d_psi = fabs(psi[check_i][check_j] - psi_old[check_i][check_j]);
            double d_zeta = fabs(zeta[check_i][check_j] - zeta_old[check_i][check_j]);
            if (max(d_psi, d_zeta) < tolerance) {
                cout << "Zbieznosc osiagnieta po " << it << " iteracjach." << endl;
                break;
            }
        }
    }


    ofstream outfile("psi_output.csv");
    if (!outfile) {
        cerr << "Błąd przy otwieraniu pliku do zapisu." << endl;
        return 1;
    }

    outfile << "x,y,psi\n";
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            outfile << x[i] << "," << y[j] << "," << psi[i][j] << "\n";
        }
    }

    outfile.close();
    cout << "Zapisano dane do psi_output.csv" << endl;


    return 0;
}
