"""Independent reference values for the test suite (scipy / numpy / lifelines).

Run from the package root:
    uv run --with scipy --with numpy --with lifelines python scripts/oracle.py
The numbers printed here are the expected values in test/*.test.ts.
"""
import numpy as np
from scipy import stats, special, optimize
from lifelines import WeibullFitter


def rry(times, failed):
    """Rank regression on Y, Benard median ranks, Johnson adjusted ranks (own implementation)."""
    order = sorted(range(len(times)), key=lambda i: (times[i], 0 if failed[i] else 1))
    n = len(times)
    mon = 0.0
    xs, ys = [], []
    for pos, i in enumerate(order):
        if not failed[i]:
            continue
        mon += (n + 1 - mon) / (1 + (n - pos))
        F = (mon - 0.3) / (n + 0.4)
        xs.append(np.log(times[i]))
        ys.append(np.log(-np.log(1 - F)))
    slope, intercept, r, _, _ = stats.linregress(xs, ys)
    return slope, np.exp(-intercept / slope), r * r


print("== 1. six-unit complete sample 16,34,53,75,93,120")
t6 = [16, 34, 53, 75, 93, 120]
print("RRY beta, eta, R2:", rry(t6, [True] * 6))
c, loc, scale = stats.weibull_min.fit(t6, floc=0)
print("MLE scipy beta, eta:", repr(c), repr(scale))
wf = WeibullFitter().fit(t6, [1] * 6)
print("lifelines rho, lambda:", repr(wf.rho_), repr(wf.lambda_), "loglik", repr(wf.log_likelihood_))

print("== 2. censored sample")
tc = [150, 340, 560, 800, 1130, 1720, 2470, 4210, 5230, 6890]
ec = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]
wf = WeibullFitter().fit(tc, ec)
print("lifelines rho, lambda:", repr(wf.rho_), repr(wf.lambda_), "loglik", repr(wf.log_likelihood_))
V = wf.variance_matrix_.values  # order lambda_, rho_
lam, rho = wf.lambda_, wf.rho_
J = np.diag([1 / lam, 1 / rho])  # delta method to (ln lambda, ln rho)
print("cov [ln eta, ln beta]:\n", repr(J @ V @ J))
print("RRY on censored:", rry(tc, [bool(e) for e in ec]))

print("== 3. distribution values beta=1.8 eta=1200")
b, e = 1.8, 1200.0
d = stats.weibull_min(b, scale=e)
for t in [100.0, 1200.0, 3000.0]:
    print(t, "pdf", repr(d.pdf(t)), "cdf", repr(d.cdf(t)), "R", repr(d.sf(t)), "h", repr(d.pdf(t) / d.sf(t)))
print("mean", repr(d.mean()), "B10", repr(d.ppf(0.10)), "B50", repr(d.ppf(0.5)), "B1", repr(d.ppf(0.01)))
print("cond R(age=500, mission=200)", repr(d.sf(700) / d.sf(500)))
for age in [0.0, 500.0, 2000.0]:
    m = optimize.brentq(lambda x: d.sf(age + x) / d.sf(age) - 0.5, 0, 1e6, xtol=1e-12)
    print("median residual life at", age, repr(m))

print("== 4. special functions")
for z in [0.3, 1.0, 2.5, 4.5, 10.0]:
    print("gamma", z, repr(special.gamma(z)), "lngamma", repr(special.gammaln(z)))
print("lngamma 171.5", repr(special.gammaln(171.5)))
for p in [1e-10, 0.001, 0.05, 0.5, 0.95, 0.975, 0.999]:
    print("norm.ppf", p, repr(stats.norm.ppf(p)))
print("betainc(2,3,0.4)", repr(special.betainc(2, 3, 0.4)), "betainc(0.5,7,0.9)", repr(special.betainc(0.5, 7, 0.9)))
print("F.ppf(0.95,1,30)", repr(stats.f.ppf(0.95, 1, 30)), "F.ppf(0.99,3,60)", repr(stats.f.ppf(0.99, 3, 60)))
k, n = 2, 60
print("T2 limit k=2 n=60 0.99", repr(k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(0.99, k, n - k)))

print("== 5. PCA on deterministic 3-feature data")
rows = []
for i in range(80):
    a = np.sin(i * 0.37) * 3
    bb = np.cos(i * 0.11) * 2
    rows.append([a + 0.1 * np.sin(i * 1.7), 2 * a + bb + 0.05 * np.cos(i * 2.3), bb - a + 0.2 * np.sin(i * 0.9 + 1)])
X = np.array(rows)
Z = (X - X.mean(0)) / X.std(0, ddof=1)
C = Z.T @ Z / (len(X) - 1)
w, v = np.linalg.eigh(C)
idx = np.argsort(w)[::-1]
w = w[idx]
v = v[:, idx]
print("eigenvalues", repr(w))
k = min(int(np.searchsorted(np.cumsum(w) / w.sum(), 0.9) + 1), 2)
P = v[:, :k]
spe = ((Z - Z @ P @ P.T) ** 2).sum(1)
print("retained", k, "spe q99", repr(np.quantile(spe, 0.99)))
x = np.array([1.0, -2.0, 4.0])
zx = (x - X.mean(0)) / X.std(0, ddof=1)
print("score spe", repr(((zx - P @ (P.T @ zx)) ** 2).sum()), "t2", repr(((P.T @ zx) ** 2 / w[:k]).sum()))
print("t2 limit", repr(k * 79 * 81 / (80 * (80 - k)) * stats.f.ppf(0.99, k, 80 - k)))
