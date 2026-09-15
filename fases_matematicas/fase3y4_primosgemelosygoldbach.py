import numpy as np
import time
from numba import njit, prange

@njit(cache=True)
def escalar_a_coordenadas_5d(n, limite_n, strides, L):
    """Mapea el número par a su coordenada espacial 5D en el Penteracto."""
    idx = n % limite_n
    x1 = (idx // strides[0]) % L
    x2 = (idx // strides[1]) % L
    x3 = (idx // strides[2]) % L
    x4 = (idx // strides[3]) % L
    x5 = idx % L
    return x1, x2, x3, x4, x5

@njit(cache=True)
def es_primo_fast(n):
    """Verificación rápida de primalidad (Criba por saltos de 6)."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if (n & 1) == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

@njit(cache=True)
def encontrar_descomposicion_goldbach(n):
    """
    Busca el par de primos (p, q) que sumados dan 'n'.
    Retorna p (el primo menor) si tiene éxito, o -1 si encuentra un contraejemplo.
    """
    # El único par que usa el 2 es 4 (2+2), el resto usa primos impares
    if n == 4:
        return 2
        
    # Barremos los candidatos impares desde p = 3 hasta n // 2
    for p in range(3, n // 2 + 1, 2):
        if es_primo_fast(p):
            q = n - p
            if es_primo_fast(q):
                return p # Éxito: Encontró la pareja (p, n-p)
                
    return -1 # ¡CONTRAEJEMPLO ENCONTRADO! (Premio del milenio conceptual)

@njit(parallel=True, cache=True)
def escanear_bloque_goldbach(inicio, fin):
    """
    Paralelismo masivo en CPU. 
    Verifica que todos los números pares en el rango cumplan la conjetura.
    """
    # Asegurar que el inicio sea par y mayor que 2
    if (inicio & 1) != 0:
        inicio += 1
    if inicio < 4:
        inicio = 4
        
    tamaño_bucle = (fin - inicio) // 2
    
    # Registros para capturar los datos de la última descomposición del bloque
    ultimo_par_procesado = 0
    ultimo_primo_p = 0
    contraejemplos_encontrados = 0

    for idx in prange(tamaño_bucle):
        n_actual = inicio + idx * 2
        p_encontrado = encontrar_descomposicion_goldbach(n_actual)
        
        if p_encontrado == -1:
            contraejemplos_encontrados += 1
            
    # Pasada rápida para extraer la última muestra del lote con fines métricos
    for idx in range(tamaño_bucle - 1, -1, -1):
        n_actual = inicio + idx * 2
        p_encontrado = encontrar_descomposicion_goldbach(n_actual)
        if p_encontrado != -1:
            ultimo_par_procesado = n_actual
            ultimo_primo_p = p_encontrado
            break
            
    return contraejemplos_encontrados, ultimo_par_procesado, ultimo_primo_p
if __name__ == "__main__":
    # Parámetros del penteracto para el mapeo de coordenadas
    LIMITE_N = 10**10
    L = 100
    STRIDES = np.array([L**4, L**3, L**2, L**1, 1], dtype=np.int64)

    # Definimos la ventana de escaneo masivo
    RANGO_INICIO = 500_000_000
    RANGO_FIN   = 502_000_000 # Evaluará todos los pares dentro de este delta de 2M (1M de pares)
    
    print("🛸 Compilando motor de descomposición de Goldbach JIT...")
    _, _, _ = escanear_bloque_goldbach(4, 10)
    print("🔥 ¡Compilación JIT completada! Preparado para la ráfaga paralela.\n")

    print(f"📊 Analizando {((RANGO_FIN - RANGO_INICIO)//2):,} números pares en la frontera de los 500M...")
    t0 = time.time()
    
    fallos, ultimo_par, primo_p = escanear_bloque_goldbach(RANGO_INICIO, RANGO_FIN)
    
    t1 = time.time()
    tiempo_total = t1 - t0

    print(f"✅ ¡Escaneo de Goldbach completado!")
    print(f"⏱️ Tiempo de procesamiento: {tiempo_total:.4f} segundos")
    print(f"🚀 Rendimiento: {((RANGO_FIN - RANGO_INICIO)//2) / tiempo_total:,.2f} números pares verificados por segundo")
    print(f"❌ Contraejemplos que rompen la conjetura hallados: {fallos}")

    if ultimo_par > 0:
        coord = escalar_a_coordenadas_5d(ultimo_par, LIMITE_N, STRIDES, L)
        print(f"\n📍 Muestra del último número par indexado con éxito:")
        print(f"   🚩 Número Par: {ultimo_par:,}")
        print(f"   🧩 Combinación Goldbach: {primo_p:,} + {ultimo_par - primo_p:,} = {ultimo_par:,}")
        print(f"   📐 Dirección en el Penteracto 5D: {coord}")
