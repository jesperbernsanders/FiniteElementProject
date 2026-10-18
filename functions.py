"""Functions shared by Stefan_notebook.ipynb and Stefan_notebook_forward_euler.ipynb."""
import numpy as np
import skfem as fem
from skfem.helpers import dot, grad
from scipy.optimize import brentq
from scipy.special import erf

from params import a_L, m_L, St


# Assembly

@fem.BilinearForm
def mass(u, v, w):  return u * v                      # M_ij = (phi_j, phi_i)
@fem.BilinearForm
def stiff(u, v, w): return dot(grad(u), grad(v))      # K_ij = (phi_j', phi_i')
@fem.BilinearForm
def adv(u, v, w):   return w.x[0] * grad(u)[0] * v    # C_ij = (y phi_j', phi_i)
@fem.LinearForm
def load(v, w):     return w.x[0] * v                 # b_i  = (y, phi_i)

def assemble(y):
    mesh = fem.MeshLine(y)
    basis = fem.Basis(mesh, fem.ElementLineP1())
    M, K, C = (form.assemble(basis) for form in (mass, stiff, adv))
    b = load.assemble(basis)
    I = mesh.interior_nodes()                         # w = 0 at y = 0 and y = 1
    return M[I][:, I], K[I][:, I], C[I][:, I], b[I]


# Analytical (Neumann) solution

lam = brentq(lambda l: np.sqrt(np.pi) * l * np.exp(l**2) * erf(l) - St, 0.0, 5.0, xtol=1e-14)

def s_exact(t):
    return 2 * lam * np.sqrt(a_L * t)

def m_exact(x, t):
    m = m_L * (1 - erf(x / (2 * np.sqrt(a_L * t))) / erf(lam))
    return np.where(x <= s_exact(t), m, 0.0)

def u_exact(y):
    return m_L * (1 - erf(lam * y) / erf(lam))    # temperature in y, independent of t


# Errors

@fem.Functional
def sq_err(w): return (w['mh'] - m_exact(w.x[0], w['t']))**2

def l2_error(x, m, t):
    # L2 error of the P1 function with nodal values m on the nodes x, against m_exact(., t)
    basis = fem.Basis(fem.MeshLine(x), fem.ElementLineP1(), intorder=6)
    return np.sqrt(sq_err.assemble(basis, mh=basis.interpolate(m), t=t))

def l2_norm(y, m):
    # L2 norm of the P1 function with nodal values m on the nodes y
    M = mass.assemble(fem.Basis(fem.MeshLine(y), fem.ElementLineP1()))
    return np.sqrt(m @ M @ m)

def eoc(err):
    # experimental order of convergence EOC_l = log(e_l / e_{l+1}) / log(2), grid refined by 2 each time
    return np.log(err[:-1] / err[1:]) / np.log(2)


# Tables

def sci_tex(e):
    # 1.234e-05 -> $1.234 \cdot 10^{-5}$
    mant, exp = f"{e:.3e}".split("e")
    return f"${mant} \\cdot 10^{{{int(exp)}}}$"

def eoc_table(path, N_name, N, err_s, err_m):
    # print the errors e_s, e_m and EOCs and save them as a booktabs tabular for \input{...} in LaTeX
    eoc_s, eoc_m = np.r_[np.nan, eoc(err_s)], np.r_[np.nan, eoc(err_m)]
    fmt = lambda x: "--" if np.isnan(x) else f"{x:.3f}"
    print(f"{N_name:>8} {'e_s':>12} {'EOC':>7} {'e_m':>12} {'EOC':>7}")
    lines = [r"\begin{tabular}{rcccc}", r"\toprule",
             f"${N_name}$ & $e_s$ & EOC & $e_m$ & EOC \\\\", r"\midrule"]
    for n, es, ps, em, pm in zip(N, err_s, eoc_s, err_m, eoc_m):
        print(f"{n:>8} {es:>12.3e} {fmt(ps):>7} {em:>12.3e} {fmt(pm):>7}")
        lines.append(f"{n} & {sci_tex(es)} & {fmt(ps)} & {sci_tex(em)} & {fmt(pm)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved to {path}")
