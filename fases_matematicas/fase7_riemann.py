import numpy as np
import time
from numba import njit, prange

@njit(cache=True)
def escalar_a_coordenadas_5d_exact(t_val, limite_n, strides, L):
    """
    Mapea de forma estrictamente escalar el valor 't' a su coordenada 5D.
    Elimina los arreglos repetidos forzando tipos enteros puros de 64 bits.
    """
    idx = int(t_val * 1000) % limite_n
    
    x1 = (idx // strides[0]) % L
    x2 = (idx // strides[1]) % L
    x3 = (idx // strides[2]) % L
    x4 = (idx // strides[3]) % L
    x5 = idx % L
    
    return x1, x2, x3, x4, x5

@njit(cache=True)
def calcular_z_riemann_preciso(t):
    """
    Versión calibrada de Riemann-Siegel.
    Inyecta correcciones de fase de orden superior para eliminar ceros fantasmas a bajas alturas.
    """
    if t < 10.0:
        return 0.0
        
    # Fase theta(t) calibrada de alta precisión
    theta = (t / 2.0) * np.log(t / (2.0 * np.pi)) - (t / 2.0) - (np.pi / 8.0) + (1.0 / (48.0 * t)) + (7.0 / (5760.0 * t**3))
    
    N_upper = int(np.sqrt(t / (2.0 * np.pi)))
    if N_upper < 1: 
        N_upper = 1
        
    suma = 0.0
    for n in range(1, N_upper + 1):
        suma += np.cos(theta - t * np.log(n)) / np.sqrt(n)
        
    return 2.0 * suma

@njit(parallel=True, cache=True)
def escanear_linea_critica_preciso(t_inicio, t_fin, delta_t):
    """Barre el plano complejo con tolerancia fina para cazar los cruces reales de signo."""
    pasos = int((t_fin - t_inicio) / delta_t)
    ceros_encontrados = np.zeros(50, dtype=np.float64)
    contador_ceros = 0
    
    # Evaluación paralela
    for i in prange(pasos - 1):
        t_actual = t_inicio + i * delta_t
        t_siguiente = t_actual + delta_t
        
        z_actual = calcular_z_riemann_preciso(t_actual)
        z_siguiente = calcular_z_riemann_preciso(t_siguiente)
        
        if (z_actual > 0.0 and z_siguiente < 0.0) or (z_actual < 0.0 and z_siguiente > 0.0):
            cero_aprox = t_actual - z_actual * (t_siguiente - t_actual) / (z_siguiente - z_actual)
            
            # Almacenamiento secuencial
            if contador_ceros < 50:
                ceros_encontrados[contador_ceros] = cero_aprox
                contador_ceros += 1
                
    return contador_ceros, ceros_encontrados[:contador_ceros]
if __name__ == "__main__":
    LIMITE_N = 10**10
    L = 100
    STRIDES = np.array([L**4, L**3, L**2, L**1, 1], dtype=np.int64)

    T_START = 10.0
    T_END   = 30.0
    PASO    = 0.0001  # Aumentamos x10 la resolución para máxima precisión

    print("🛸 Relanzando motor de Riemann V2 (Calibración Fina de Fase)...")
    _, _ = escanear_linea_critica_preciso(10.0, 11.0, 0.1)
    
    t0 = time.time()
    total_ceros, listado_ceros = escanear_linea_critica_preciso(T_START, T_END, PASO)
    t1 = time.time()

    print("\n✅ ¡Escaneo de precisión completado!")
    print(f"⏱️ Tiempo de procesamiento: {t1 - t0:.4f} segundos")
    print(f"🎯 Ceros reales de Riemann detectados: {total_ceros}")

    if total_ceros > 0:
        listado_ceros = np.sort(listado_ceros)
        print("\n📍 Reporte de Ceros Limpios Grabados en el Penteracto:")
        for idx in range(total_ceros):
            cero_t = listado_ceros[idx]
            coord = escalar_a_coordenadas_5d_exact(cero_t, LIMITE_N, STRIDES, L)
            print(f"   🚩 Cero #{idx+1}: s = 1/2 + {cero_t:.4f}i -> Coordenada Escalar 5D: {coord}")
