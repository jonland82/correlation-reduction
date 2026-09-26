"""Reproduces the equilibrium and steady-state values. Run: python source/extensions_calculations.py"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov as lyap
from scipy.optimize import brentq

kB = 1.380649e-23; hbar = 1.054571817e-34
w0 = 2*np.pi*400e3
Ls = np.array([6, 8, 58, 85.0])
T = 300.0

def I_gauss(C, n1):
    """Mutual information (nats) of a zero-mean Gaussian with covariance C; first n1 coords = subsystem A."""
    A = C[:n1, :n1]; B = C[n1:, n1:]
    return 0.5*np.log(np.linalg.det(A)*np.linalg.det(B)/np.linalg.det(C))

print("=== 1a. Corrected bridge q = 2 Lambda / w0 ===")
for tag, fac in (("old q=L/w0", 1), ("new q=2L/w0", 2)):
    q = fac*Ls/400e3
    I = -0.5*np.log(1-q**2)
    print(tag, "q:", q, "I:", I)
    W = kB*T*(I[3]-I[2])
    print("   85->58 Hz: dI =", I[3]-I[2], " W_rev =", W, "J;  % drop", 100*(1-I[2]/I[3]))

print("\n=== 1b'. Decomposition of equilibrium dF(g) (classical, units kT=k=1) ===")
for q in (0.1, 4.25e-4):
    s = 1/(1-q**2)
    local = s-1-np.log(s)                  # two marginals' relative entropy to g=0 Gibbs
    corr = -0.5*np.log(1-q**2)             # kT I
    inter = -q**2/(1-q**2)                 # <V> = g<xy>
    total = 0.5*np.log(1-q**2)             # F(g)-F(0)
    print(f"q={q}: local={local:.3e} corr={corr:.3e} interaction={inter:.3e} sum={local+corr+inter:.3e} dF={total:.3e} -kTI={-corr:.3e}")

print("\n=== 1b. Two-bath NESS (Yang params) ===")
g1, g2 = 2*np.pi*12, 2*np.pi*6

def ness_rwa(L, n1, n2):
    """Closed-form RWA steady state. Returns N11, N22, Im<a1 a2*>."""
    G = g1+g2
    s = 2*L*g1*g2*(n1-n2)/(G*(g1*g2+4*L**2))
    return n1-2*L*s/g1, n2+2*L*s/g2, s

def ness_full(L, T1, T2):
    """Full (non-RWA) classical Langevin, lab frame. State (x1,v1,x2,v2), m=1.
    Yang Eq.1 in lab-frame form: stiffness k+g on each, cross-stiffness g, g = 2 w0 L (Lambda<0 => negative spring)."""
    # dimensionless: time in 1/w0, m=1, kB=1 -> k=1, rates divided by w0
    l, a1, a2 = L/w0, g1/w0, g2/w0
    g = 2*l
    A = np.array([[0, 1, 0, 0],
                  [-(1+g), -a1, -g, 0],
                  [0, 0, 0, 1],
                  [-g, 0, -(1+g), -a2]])
    D = np.diag([0, 2*a1*T1, 0, 2*a2*T2])
    C = lyap(A, -D)
    res = A@C + C@A.T + D
    assert np.abs(res).max() < 1e-9*np.abs(D).max(), np.abs(res).max()
    return C

for ratio in (2.0, 10.0):
    TH, TL = ratio, 1.0     # mode 1 = hot (gamma1), as in Yang Fig. 1
    print(f"-- T_H/T_L = {ratio}")
    for L0 in Ls:
        if True:
            L = -2*np.pi*L0   # Yang: effective spring negative (sign does not affect I)
            N11, N22, s = ness_rwa(L, TH, TL)
            rho_c = abs(s)/np.sqrt(N11*N22)
            I_ps_rwa = -np.log(1-rho_c**2)
            C = ness_full(L, TH, TL)
            I_ps = I_gauss(C, 2)
            Cx = C[np.ix_([0, 2], [0, 2])]
            I_pos = I_gauss(Cx, 1)
            rho_x = Cx[0, 1]/np.sqrt(Cx[0, 0]*Cx[1, 1])
            rho_xv = C[0, 3]/np.sqrt(C[0, 0]*C[3, 3])
            # equilibrium toy at same q:
            q = 2*abs(L)/w0
            I_eq = -0.5*np.log(1-q**2)
            # heat flux check: J = gamma2*(E2 - kT_L) with E2 = m v2^2 (units kB=1)
            J_full = g2*(C[3, 3]-TL)   # velocity variance = kT_eff (m=1)
            J_lim = g1*g2*(TH-TL)/(g1+g2)
            print(f"  L/2pi={L0:5.0f} Hz  Teff1={N11:.3f} Teff2={N22:.3f}  |corr|={rho_c:.4f}  "
                  f"I_phase(RWA)={I_ps_rwa:.3e} I_phase(full)={I_ps:.3e}  I_pos(full)={I_pos:.3e} rho_x={rho_x:.2e} "
                  f"rho_x1v2={rho_xv:.3f} | eq toy I={I_eq:.2e} | J/Jmax={J_full/J_lim:.3f}")
# equal temperature sanity: full model must reproduce Gibbs rho = -g/(k+g)
C = ness_full(-2*np.pi*85, 1.0, 1.0)
g = 2*(-2*np.pi*85)/w0; print("equal-T check rho_x:", C[0, 2]/C[0, 0], "Gibbs:", -g/(1+g), " x-v2 corr:", C[0, 3])

print("\n=== 2a. Quantum Gibbs state of position-coupled oscillators ===")
def gibbs_quantum(q, tau):
    """Units hbar=m=w0=1; tau = kT/(hbar w0). Normal modes w+-=sqrt(1+-q).
    Returns F(q)-F(0), I, E_N (nats)."""
    wp, wm = np.sqrt(1+q), np.sqrt(1-q)
    def coth(x): return 1/np.tanh(x)
    cp, cm = coth(wp/(2*tau)), coth(wm/(2*tau))
    # covariances (hbar=1): normal modes (x+-y)/sqrt2
    Vx = 0.25*(cp/wp + cm/wm); Cx = 0.25*(cp/wp - cm/wm)
    Vp = 0.25*(cp*wp + cm*wm); Cp = 0.25*(cp*wp - cm*wm)
    def S(nu):  # entropy of mode with symplectic eigenvalue nu (vacuum=1/2)
        a, b = nu+0.5, nu-0.5
        return a*np.log(a) - (b*np.log(b) if b > 1e-300 else 0.0)
    Sglob = S(0.5*cp) + S(0.5*cm)
    Sloc = S(np.sqrt(Vx*Vp))
    I = 2*Sloc - Sglob
    nus = [np.sqrt((Vx+Cx)*(Vp-Cp)), np.sqrt((Vx-Cx)*(Vp+Cp))]
    EN = max(0.0, -np.log(2*min(nus)))
    def F(w): return w/2 + tau*np.log1p(-np.exp(-w/tau))
    dF = F(wp)+F(wm)-2*F(1.0)
    return dF, I, EN

for q in (0.2, 4.25e-4):
    print(f"q={q}")
    for tau in (1e-3, 0.05, 0.1, 0.3, 1, 10, 1e3):
        dF, I, EN = gibbs_quantum(q, tau)
        print(f"  kT/hw={tau:8.3g}  -dF(decouple cost)={-dF:.4e}  kT*I={tau*I:.4e}  ratio={-dF/(tau*I):.4f}  E_N={EN:.3e}")
    # threshold temperature for entanglement
    # entangled iff coth(w+/2kT) coth(w-/2kT) < w+/w-
    wp, wm = np.sqrt(1+q), np.sqrt(1-q)
    tc = brentq(lambda t: 1/np.tanh(wp/(2*t))/np.tanh(wm/(2*t)) - wp/wm, 1e-3, 5)
    print(f"  entanglement vanishes at kT/hw ~ {tc:.4f}; approx 1/ln(4/q) = {1/np.log(4/q):.4f}")
    if q < 1e-3:
        print("  -> T_c =", tc*hbar*w0/kB*1e6, "microkelvin;  n_th at 300 K =", kB*300/(hbar*w0))
dF0, I0, EN0 = gibbs_quantum(0.2, 1e-6)
print("T=0, q=0.2: ground-energy cost", -dF0, " E_N", EN0, " approx q/2", 0.1, " cost/E_N^2", -dF0/EN0**2)

print("\n=== 2b. Two-mode squeezed thermal state, r=0.5 ===")
def g_ent(n): return (n+1)*np.log(n+1) - (n*np.log(n) if n > 0 else 0)
r = 0.5
for nth in (0, 0.1, 0.5, 0.859, 1, 5, 100):
    na = ((2*nth+1)*np.cosh(2*r)-1)/2
    I = 2*g_ent(na)-2*g_ent(nth)
    EN = max(0.0, 2*r-np.log(2*nth+1))
    W = -2*(2*nth+1)*np.sinh(r)**2      # W_on for unitary unsqueezing, units hbar w
    print(f"  n_th={nth:6}: I={I:.4f}  E_N={EN:.4f}  W_on={W:.4f} hbar w")
print("  large-n limit of I = 2 ln cosh 2r =", 2*np.log(np.cosh(2*r)))
print("  E_N=0 threshold n_th = (e^{2r}-1)/2 =", (np.exp(2*r)-1)/2)
