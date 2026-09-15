import numpy as np
import time
from numba import njit, prange

@njit(cache=True)
def escalar_a_coordenadas_5d(c, limite_n, strides, L):
    """Mapea el valor de 'c' a su coordenada 5D en el Penteracto."""
    idx = c % limite_n
    x1 = (idx // strides) % L
    x2 = (idx // strides) % L
    x3 = (idx // strides) % L
    x4 = (idx // strides) % L
    x5 = idx % L
    return x1, x2, x3, x4, x5

@njit(cache=True)
def mcd_fast(a, b):
    """Algoritmo de Euclides binario ultra rápido para verificar coprimalidad."""
    if a == 0: return b
    if b == 0: return a
    
    # Encontrar factor común de 2
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1
        
    while (a & 1) == 0:
        a >>= 1
        
    while b != 0:
        while (b & 1) == 0:
            b >>= 1
        if a > b:
            # Intercambiar
            temp = a
            a = b
            b = temp
        b -= a
        
    return a << shift

@njit(cache=True)
def calcular_radical(n):
    """Calcula el producto de los factores primos únicos de un número."""
    if n <= 1:
        return 1
    
    rad = 1
    temp = n
    
    # Evaluar factor 2
    if (temp & 1) == 0:
        rad *= 2
        while (temp & 1) == 0:
            temp >>= 1
            
    # Evaluar factores impares
    d = 3
    while d * d <= temp:
        if temp % d == 0:
            rad *= d
            while temp % d == 0:
                temp //= d
        d += 2
        
    if temp > 1:
        rad *= temp
        
    return rad

@njit(cache=True)
def evaluar_tripleta_abc(a, b):
    """
    Evalúa la tripleta (a, b, c) donde a + b = c.
    Retorna el radical si califica como un golpe o anomalía potencial (c >= rad).
    De lo contrario, retorna 0.
    """
    c = a + b
    
    # Filtro 1: Deben ser coprimos estrictos
    if mcd_fast(a, b) != 1:
        return 0
        
    # Filtro 2: Calcular radical combinado de forma eficiente
    # rad(abc) = rad(a) * rad(b) * rad(c) si son coprimos
    rad_abc = calcular_radical(a) * calcular_radical(b) * calcular_radical(c)
    
    # ¿Encontramos una anomalía de calidad abc? (c > rad(abc))
    if c > rad_abc:
        return rad_abc
        
    return 0

@njit(parallel=True, cache=True)
def escanear_espacio_abc(c_fijo, max_a):
    """
    Paralelismo masivo en CPU.
    Para un valor de 'c' dado, barre todos los valores posibles de 'a' (y b = c - a) 
    buscando configuraciones que desafíen la conjetura.
    """
    anomalias_locales_a = np.zeros(20, dtype=np.int64)
    anomalias_locales_rad = np.zeros(20, dtype=np.int64)
    contador = 0
    
    for a in prange(1, max_a):
        b = c_fijo - a
        if b <= a: 
            continue
            
        rad = evaluar_tripleta_abc(a, b)
        if rad > 0:
            # Guardado local seguro
            if contador < 20:
                anomalias_locales_a[contador] = a
                anomalias_locales_rad[contador] = rad
                contador += 1
                
    return contador, anomalias_locales_a[:contador], anomalias_locales_rad[:contador]
if __name__ == "__main__":
    # Dimensiones lógicas del penteracto
    LIMITE_N = 10**10
    L = 100
    STRIDES = np.array([L**4, L**3, L**2, L**1, 1], dtype=np.int64)

    # Fijamos un valor de 'c' de prueba (ej. 243 para validar el motor numérico)
    C_OBJETIVO = 243
    
    print("🛸 Compilando motor de factorización y coprimalidad ABC JIT...")
    _, _, _ = escanear_espacio_abc(10, 5)
    print("🔥 ¡Compilación JIT exitosa! Desplegando rastreador en paralelo.\n")

    print(f"📊 Buscando combinaciones anómalas abc para c = {C_OBJETIVO}...")
    t0 = time.time()
    
    total_hallados, lista_a, lista_rad = escanear_espacio_abc(C_OBJETIVO, C_OBJETIVO)
    
    t1 = time.time()
    tiempo_total = t1 - t0

    print(f"✅ ¡Escaneo combinatorio completado!")
    print(f"⏱️ Tiempo de procesamiento: {tiempo_total:.4f} segundos")
    print(f"🎯 Anomalías abc detectadas en este nodo: {total_hallados}")

    if total_hallados > 0:
        print("\n📍 Reporte de Tripleta Anómala Indexada en el Penteracto:")
        for idx in range(total_hallados):
            val_a = lista_a[idx]
            val_b = C_OBJETIVO - val_a
            val_rad = lista_rad[idx]
            calidad = np.log(C_OBJETIVO) / np.log(val_rad)
            coord = escalar_a_coordenadas_5d(C_OBJETIVO, LIMITE_N, STRIDES, L)
            
            print(f"   🚩 Tripleta: {val_a} + {val_b} = {C_OBJETIVO}")
            print(f"   🧩 rad(abc) = {val_rad}  <--- (Cumple c > rad!)")
            print(f"   📈 Factor de Calidad (q): {calidad:.4f}")
            print(f"   📐 Dirección en el Penteracto 5D: {coord}")
