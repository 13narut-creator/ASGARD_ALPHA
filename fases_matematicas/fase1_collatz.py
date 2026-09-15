import numpy as np
import time
from numba import njit, prange

# 1. Configuración Estructural del Mega-Penteracto
LIMITE_N = 10**10
L = 100
STRIDES = np.array([L**4, L**3, L**2, L**1, 1], dtype=np.int64)

# Tamaño de la Tabla de Caché (20 Millones de elementos)
TAM_CACHE = 20_000_000

@njit(cache=True)
def llenar_cache_secuencial(cache_pasos, tam_cache):
    """Llena la caché pasándola como argumento para que sea mutable por Numba."""
    for n in range(2, tam_cache):
        actual = n
        pasos = 0
        while actual >= n and actual > 1:
            if (actual & 1) == 0:
                actual = actual >> 1
            else:
                actual = 3 * actual + 1
            pasos += 1
        
        if actual < n:
            cache_pasos[n] = pasos + cache_pasos[actual]
        else:
            cache_pasos[n] = pasos

@njit(cache=True)
def n_a_coordenadas_5d_fast(n, limite_n, strides, l_size):
    """Mapea instantáneamente un número escalar a su coordenada espacial 5D."""
    idx = n % limite_n
    x1 = (idx // strides[0]) % l_size
    x2 = (idx // strides[1]) % l_size
    x3 = (idx // strides[2]) % l_size
    x4 = (idx // strides[3]) % l_size
    x5 = idx % l_size
    return x1, x2, x3, x4, x5

@njit(cache=True)
def simular_trayectoria_con_cache(n_inicial, cache_pasos, tam_cache):
    """Calcula los pasos de Collatz usando atajos de memoria estática pasados por argumento."""
    n = n_inicial
    pasos = 0
    
    while n > 1:
        if n < tam_cache:
            return pasos + cache_pasos[n]
            
        if (n & 1) == 0:
            n = n >> 1
        else:
            n = 3 * n + 1
        pasos += 1
        
        if pasos > 20000:  # Límite de seguridad académica
            return pasos
            
    return pasos

@njit(parallel=True, cache=True)
def procesar_bloque_con_filtro(inicio, fin, base_record, cache_pasos, tam_cache):
    """
    Paralelismo masivo en CPU. 
    Filtra y captura en tiempo real los récords sin colisiones globales.
    """
    tamaño = fin - inicio
    pasos_locales = np.zeros(tamaño, dtype=np.int32)
    
    # Estructuras para almacenar candidatos récord encontrados en este bloque (Máx 1000)
    indices_record = np.zeros(1000, dtype=np.int64)
    valores_record = np.zeros(1000, dtype=np.int32)
    contador_records = 0
    
    record_actual = base_record

    for i in prange(tamaño):
        n_actual = inicio + i
        p = simular_trayectoria_con_cache(n_actual, cache_pasos, tam_cache)
        pasos_locales[i] = p
        
        if p > record_actual:
            record_actual = p
            
    # Consolidación secuencial segura del bloque
    for i in range(tamaño):
        if pasos_locales[i] > base_record:
            if contador_records < 1000:
                indices_record[contador_records] = inicio + i
                valores_record[contador_records] = pasos_locales[i]
                base_record = pasos_locales[i]  
                contador_records += 1
            else:
                break
                
    return pasos_locales, indices_record[:contador_records], valores_record[:contador_records]

# --- SUBSISTEMA DE ARRANQUE CORREGIDO ---
if __name__ == "__main__":
    # Creamos el arreglo mutable en el ámbito principal (__main__)
    CACHE_PASOS = np.zeros(TAM_CACHE, dtype=np.int32)

    print("🛸 Generando e inyectando Look-Up Table de 20M en la caché de CPU...")
    t_start = time.time()
    llenar_cache_secuencial(CACHE_PASOS, TAM_CACHE)
    print(f"⚡ ¡Caché lista en {time.time() - t_start:.2f} segundos!")

    # Filtro base: Ponemos 500 para forzar capturas en la ráfaga de prueba
    RECORD_HISTORICO_BASE = 500  
    
    RANGO_INICIO = 50_000_000
    RANGO_FIN = 55_000_000
    LOTE_SIZE = RANGO_FIN - RANGO_INICIO
    
    print(f"\n📊 Analizando ráfaga de {LOTE_SIZE:,} números en el Penteracto 5D con Filtro de Récords...")
    
    t0 = time.time()
    pasos, numeros_top, valores_top = procesar_bloque_con_filtro(
        RANGO_INICIO, RANGO_FIN, RECORD_HISTORICO_BASE, CACHE_PASOS, TAM_CACHE
    )
    t1 = time.time()
    
    tiempo_total = t1 - t0
    
    print(f"✅ ¡Simulación e inspección completas!")
    print(f"⏱️ Tiempo de ejecución: {tiempo_total:.4f} segundos")
    print(f"🚀 Rendimiento optimizado: {LOTE_SIZE / tiempo_total:,.2f} números/segundo")
    
    # 3. Reporte de Hallazgos
    print(f"\n🎯 [REPORTE FILTRADO DE ANOMALÍAS (Pasos > {RECORD_HISTORICO_BASE})]:")
    if len(numeros_top) == 0:
        print("   No se encontraron nuevos picos en este rango específico.")
    else:
        print(f"   Se detectaron {len(numeros_top)} números que superaron la barrera base.")
        top_indices = np.argsort(valores_top)[::-1][:5]
        for idx in top_indices:
            n_anomalo = numeros_top[idx]
            pasos_anomalos = valores_top[idx]
            coord_5d = n_a_coordenadas_5d_fast(n_anomalo, LIMITE_N, STRIDES, L)
            print(f"   🚩 Número: {n_anomalo:,} | Pasos: {pasos_anomalos} -> Coordenada Penteracto: {coord_5d}")
