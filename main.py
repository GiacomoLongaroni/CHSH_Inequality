from BasisMeasurement import BasisMeasurement

def S_routine():
    import numpy as np
    E = np.zeros((2, 2))
    E_err = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            print(f"Loading basis A{i},B{j}", end='')
            basis = BasisMeasurement(i, j)
            E[i, j] = basis.E
            E_err[i, j] = basis.E_err
            print(f" -> E = {E[i,j]:.3f} ± {E_err[i,j]:.3f}")
    S = np.sum(np.abs(E))
    S_err = np.sqrt(np.sum(E_err**2))
    print(f"\nS: {S:.3f} ± {S_err:.3f}")

if __name__ == "__main__":
    S_routine()