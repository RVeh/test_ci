from dataclasses import dataclass

# ========== 1. Modellparameter ========== 
@dataclass
class CIRealConfig:
    h: float
    n: int
    gamma: float = 0.95

# ========================================

# ======== 2. Intervall-Methoden ========= 
from math import sqrt
from statistics import NormalDist
from scipy.stats import beta

def z_value(gamma: float) -> float:
    alpha = 1.0 - gamma
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)

def wald_ci(h, n, gamma):
    z = z_value(gamma)
    half = z * sqrt(h * (1 - h) / n)
    return h - half, h + half

def wilson_ci(h, n, gamma):
    z = z_value(gamma)
    denom = 1 + z**2 / n
    center = (h + z**2 / (2*n)) / denom
    radius = z * sqrt((h * (1 - h) + z**2 / (4*n)) / n) / denom
    return center - radius, center + radius

def clopper_pearson_ci(h, n, gamma):
    k = round(h * n)
    alpha = 1.0 - gamma
    L = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    R = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return L, R

# ========================================

# ======== 3. Ausgabe-Funktion =========== 
def ci_realisations(cfg: CIRealConfig):
    return {
        "Wald-KI": wald_ci(cfg.h, cfg.n, cfg.gamma),
        "Wilson_KI": wilson_ci(cfg.h, cfg.n, cfg.gamma),
        "Clopper-Pearson-KI": clopper_pearson_ci(cfg.h, cfg.n, cfg.gamma)
    }
    
# ========================================