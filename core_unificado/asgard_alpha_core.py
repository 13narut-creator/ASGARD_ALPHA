import numpy as np
import time
from numba import njit, prange

# ============================================================
# CONFIGURACIÓN PARAMÉTRICA (ENTORNO DE TESIS)
# ============================================================
LIMITE_N = 10**10
L_SIZE = 100
STRIDES = np.array([L_SIZE**4, L_SIZE**3, L_SIZE**2, L_SIZE**1, 1], dtype=np.int64)

@njit(cache=True)
def escalar_a_coordenadas_5d_exact(scalar_val, limite_n, strides, l_size):
    """Mapeo geométrico implícito de variables continuas a la red 5D."""
    idx = int(scalar_val * 1000) % limite_n
    x1 = (idx // strides) % l_size
    x2 = (idx // strides) % l_size
    x3 = (idx // strides) % l_size
    x4 = (idx // strides) % l_size
    x5 = idx % l_size
    return x1, x2, x3, x4, x5

@njit(parallel=True, cache=True)
def calcular_tensores_nodos(pos_nodos, indices_vecinos, pesos_enlaces, espines):
    """
    MÓDULO 1: CÁLCULO DEL TENSOR MÉTRICO GENUINO (g_mu_nu)
    Genera las 10 componentes independientes por cada nodo a partir del grafo.
    """
    n_nodos = len(pos_nodos)
    g_tensor = np.zeros((n_nodos, 10), dtype=np.float64)
    
    for i in prange(n_nodos):
        # Componente Temporal g00 = -1 - 2phi
        suma_diff_espines = 0.0
        vecinos = indices_vecinos[i]
        for v_idx, v in enumerate(vecinos):
            suma_diff_espines += pesos_enlaces[i, v_idx] * (espines[i] - espines[v])**2
            
        potencial_phi = np.maximum(suma_diff_espines, 1e-6)
        g_tensor[i, 0] = -1.0 - 2.0 * potencial_phi
        
        # Componentes Espaciales (Anisotropía de Aristas)
        for v_idx, v in enumerate(vecinos):
            w = pesos_enlaces[i, v_idx]
            dx = pos_nodos[v, 0] - pos_nodos[i, 0]
            dy = pos_nodos[v, 1] - pos_nodos[i, 1]
            dz = pos_nodos[v, 2] - pos_nodos[i, 2]
            
            g_tensor[i, 1] += w * dx * dx  # g11
            g_tensor[i, 2] += w * dy * dy  # g22
            g_tensor[i, 3] += w * dz * dz  # g33
            g_tensor[i, 4] += w * dx * dy  # g01
            g_tensor[i, 5] += w * dx * dz  # g02
            g_tensor[i, 6] += w * dy * dz  # g03
            g_tensor[i, 7] += w * dx * dy  # g12
            g_tensor[i, 8] += w * dx * dz  # g13
            g_tensor[i, 9] += w * dy * dz  # g23
            
        g_tensor[i, 1] = 1.0 + g_tensor[i, 1] * 0.01
        g_tensor[i, 2] = 1.0 + g_tensor[i, 2] * 0.01
        g_tensor[i, 3] = 1.0 + g_tensor[i, 3] * 0.01
        
    return g_tensor

@njit(parallel=True, cache=True)
def evaluar_fuerzas_geodesicas(pos_p, g_tensor, pos_nodos, indices_vecinos):
    """Interpola los gradientes del tensor para resolver la ecuación de la geodésica."""
    n_part = len(pos_p)
    fuerzas = np.zeros((n_part, 3), dtype=np.float64)
    
    for p in prange(n_part):
        p_pos = pos_p[p]
        min_dist = 1e10
        nodo_cercano = 0
        
        for n in range(len(pos_nodos)):
            dx = pos_nodos[n, 0] - p_pos[:, 0] if pos_p.ndim > 1 else pos_nodos[n, 0] - p_pos[0]
            dy = pos_nodos[n, 1] - p_pos[:, 1] if pos_p.ndim > 1 else pos_nodos[n, 1] - p_pos[1]
            dz = pos_nodos[n, 2] - p_pos[:, 2] if pos_p.ndim > 1 else pos_nodos[n, 2] - p_pos[2]
            d = dx*dx + dy*dy + dz*dz
            if d < min_dist:
                min_dist = d
                nodo_cercano = n
                
        vecinos = indices_vecinos[nodo_cercano]
        for v in vecinos:
            r_vec = pos_nodos[v] - pos_nodos[nodo_cercano]
            dist = np.sqrt(r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2) + 1e-5
            delta_g00 = g_tensor[v, 0] - g_tensor[nodo_cercano, 0]
            
            fuerzas[p, 0] -= 0.5 * delta_g00 * (r_vec[0] / dist)
            fuerzas[p, 1] -= 0.5 * delta_g00 * (r_vec[1] / dist)
            fuerzas[p, 2] -= 0.5 * delta_g00 * (r_vec[2] / dist)
            
    return fuerzas

@njit(parallel=True, cache=True)
def avanzar_materia_leapfrog(pos_p, vel_p, fuerzas_p, dt, limite_r):
    """
    MÓDULO 2: INTEGRACIÓN SIMPLÉCTICA BARIÓNICA
    Mueve los millones de partículas conservando el volumen del espacio de fases.
    """
    n_part = len(pos_p)
    for p in prange(n_part):
        vel_p[p, 0] += fuerzas_p[p, 0] * dt
        vel_p[p, 1] += fuerzas_p[p, 1] * dt
        vel_p[p, 2] += fuerzas_p[p, 2] * dt
        
        pos_p[p, 0] += vel_p[p, 0] * dt
        pos_p[p, 1] += vel_p[p, 1] * dt
        pos_p[p, 2] += vel_p[p, 2] * dt
        
        r_sq = pos_p[p, 0]**2 + pos_p[p, 1]**2 + pos_p[p, 2]**2
        if r_sq > limite_r**2:
            factor = limite_r / np.sqrt(r_sq)
            pos_p[p, 0] *= factor
            pos_p[p, 1] *= factor
            pos_p[p, 2] *= factor
            vel_p[p, 0] *= -0.2
            vel_p[p, 1] *= -0.2
            vel_p[p, 2] *= -0.2
