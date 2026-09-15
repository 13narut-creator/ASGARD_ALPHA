@njit(cache=True)
def explorar_vecindario_5d(nodo_inicial, n, max_pasos_caminata):
    """
    Caminata metaheurística por el penteracto usando Recocido Simulado.
    Permite aceptar mutaciones desfavorables controladas para escapar de mínimos locales.
    """
    matriz_actual = inicializar_matriz_nodo(nodo_inicial, n)
    error_actual = calcular_error_hadamard(matriz_actual, n)
    
    # Clon de respaldo para la mejor matriz histórica del hilo
    mejor_matriz = matriz_actual.copy()
    mejor_error = error_actual
    
    state = nodo_inicial ^ 0x5DEECE66D
    
    # Parámetros térmicos de recocido
    temperatura = 50.0
    factor_enfriamiento = 0.992
    
    for paso in range(max_pasos_caminata):
        if mejor_error == 0:
            return 0, mejor_matriz
            
        # Evolución del LCG de alta entropía para las coordenadas
        state = (state * 1103515245 + 12345) & 0x7fffffff
        i_mutar = (state >> 16) % n
        
        state = (state * 1103515245 + 12345) & 0x7fffffff
        j_mutar = (state >> 16) % n
        
        # Ejecutar bit-flip (mutación)
        matriz_actual[i_mutar, j_mutar] *= -1
        nuevo_error = calcular_error_hadamard(matriz_actual, n)
        
        delta_error = nuevo_error - error_actual
        
        # Criterio de Aceptación Térmica (Metrópolis)
        # Si el error disminuye, se acepta siempre (delta_error <= 0)
        # Si el error aumenta, hay una probabilidad basada en la temperatura
        state = (state * 1103515245 + 12345) & 0x7fffffff
        probabilidad_aceptacion = (state >> 16) % 1000 / 1000.0
        
        # Aproximación lineal ultrarrápida para Numba en lugar de math.exp
        umbral_aceptacion = 0.0
        if temperatura > 0.1:
            umbral_aceptacion = 1.0 - (delta_error / temperatura)
            
        if delta_error <= 0 or probabilidad_aceptacion < umbral_aceptacion:
            # Aceptamos el cambio de estado en el penteracto
            error_actual = nuevo_error
            # Registrar si es un récord absoluto para este explorador
            if error_actual < mejor_error:
                mejor_error = error_actual
                mejor_matrix = matriz_actual.copy()
        else:
            # Rechazar mutación y revertir bit-flip
            matriz_actual[i_mutar, j_mutar] *= -1
            
        # Enfriamiento gradual del sistema
        temperatura *= factor_enfriamiento
            
    return mejor_error, mejor_matriz
if __name__ == "__main__":
    ORDEN = 16            
    EXPLORADORES = 50000  
    PASOS_CAMINATA = 500  

    print(f"🔥 [FASE 2 - ANNEALING ACTIVADO] Atacando el espacio 2^256 con escape térmico...")
    print(f"🛸 Desplegando {EXPLORADORES:,} exploradores con fluctuación cuántica...")
    t0 = time.time()
    
    errores, matrices = ejecutar_busqueda_masiva(EXPLORADORES, ORDEN, PASOS_CAMINATA)
    
    t1 = time.time()
    
    print("\n✅ Simulación térmica completada con éxito!")
    print(f"⏱️ Tiempo de procesamiento: {t1 - t0:.4f} segundos")
    
    soluciones_perfectas = np.sum(errores == 0)
    print(f"🎯 Matrices de Hadamard de orden 16 encontradas: {soluciones_perfectas}")
    print(f"📉 Error mínimo absoluto alcanzado en la red: {np.min(errores)}")
    
    if soluciones_perfectas > 0:
        idx_exito = np.where(errores == 0)[0][0]
        print(f"\n📍 ¡Éxito! Matriz de Hadamard de Orden 16 Encontrada:")
        print(matrices[idx_exito])
