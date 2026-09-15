import numpy as np
import time
from numba import njit, prange

@njit(cache=True)
def escalar_a_coordenadas_5d(n, limite_n, strides, L):
    """Mapea el primer primo validador a su coordenada 5D en el Penteracto."""
    idx = n % limite_n
    x1 = (idx // strides) % L
    x2 = (idx // strides) % L
    x3 = (idx // strides) % L
    x4 = (idx // strides) % L
    x5 = idx % L
    return x1, x2, x3, x4, x5

@njit(cache=True)
def es_primo_fast(n):
    """Verificación de primalidad ultra veloz con saltos de 6."""
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
def verificar_intervalo_legendre(n):
    """
    Busca el primer número primo dentro del intervalo cuadrático [n^2, (n+1)^2].
    Retorna el primo hallado. Si fallara (contraejemplo), retornaría -1.
    """
    inicio_intervalo = n * n
    fin_intervalo = (n + 1) * (n + 1)
    
    # Asegurar arrancar en el primer número impar mayor que n^2
    candidato = inicio_intervalo + 1
    if (candidato & 1) == 0:
        candidato += 1
        
    # Buscar de forma secuencial rápida dentro de la ventana del intervalo
    while candidato < fin_intervalo:
        if es_primo_fast(candidato):
            return candidato # Éxito: retorna el primer primo validador hallado
        candidato += 2
        
    return -1 # ¡CONTRAEJEMPLO DETECTADO!

@njit(parallel=True, cache=True)
def ejecutar_escaneo_legendre(inicio_n, fin_n):
    """
    Paralelismo masivo en CPU. 
    Verifica de manera simultánea miles de intervalos cuadráticos de Legendre.
    """
    tamaño = fin_n - inicio_n
    primos_validadores = np.zeros(tamaño, dtype=np.int64)
    fallos = 0
    
    for i in prange(tamaño):
        n_actual = inicio_n + i
        primo_hallado = verificar_intervalo_legendre(n_actual)
        
        if primo_hallado == -1:
            fallos += 1
        else:
            primos_validadores[i] = primo_hallado
            
    return fallos, primos_validadores
if __name__ == "__main__":
    # Dimensiones lógicas del penteracto
    LIMITE_N = 10**10
    L = 100
    STRIDES = np.array([L**4, L**3, L**2, L**1, 1], dtype=np.int64)

    # Rango de enteros base 'n' a evaluar
    N_INICIO = 30_000
    N_FIN    = 30_500  # Evaluará 500 intervalos de Legendre de alta densidad
    INTERVALOS_TOTALES = N_FIN - N_INICIO

    print("🛸 Compilando motor JIT de Legendre e inicializando hilos...")
    _, _ = ejecutar_escaneo_legendre(2, 5)
    print("🔥 ¡Compilación exitosa! Código nativo listo para el test de estrés.\n")

    print(f"📊 Evaluando {INTERVALOS_TOTALES:,} intervalos de Legendre en la frontera alta (n² > 900M)...")
    t0 = time.time()
    
    contraejemplos, listado_primos = ejecutar_escaneo_legendre(N_INICIO, N_FIN)
    
    t1 = time.time()
    tiempo_total = t1 - t0

    print(f"✅ ¡Simulación de Legendre completada!")
    print(f"⏱️ Tiempo de procesamiento: {tiempo_total:.4f} segundos")
    print(f"🚀 Rendimiento: {INTERVALOS_TOTALES / tiempo_total:,.2f} intervalos validados por segundo")
    print(f"❌ Contraejemplos que rompen la conjetura hallados: {contraejemplos}")

    if len(listado_primos) > 0 and contraejemplos == 0:
        # Extraer los datos del último intervalo procesado satisfactoriamente
        ultimo_idx = INTERVALOS_TOTALES - 1
        n_evaluado = N_FIN - 1
        primo_validador = listado_primos[ultimo_idx]
        coord = escalar_a_coordenadas_5d(primo_validador, LIMITE_N, STRIDES, L)
        
        print(f"\n📍 Análisis del último intervalo examinado (n = {n_evaluado:,}):")
        print(f"   🚩 Ventana Cuadrática: [{n_evaluado**2:,}  <---  p  --->  {(n_evaluado+1)**2:,}]")
        print(f"   🧩 Primer Primo Hallado (p): {primo_validador:,}")
        print(f"   📐 Coordenada de Indexación 5D: {coord}")
