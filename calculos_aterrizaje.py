import numpy as np


def cargar_datos():
    """Carga los arrays de telemetría y parámetros de partida del aterrizaje."""
    # Datos para el Sistema Lineal (Geometría del tren y pesos)
    # [x_cg, peso_total, momentos en x e y]
    params_masa = np.array([18.5, 75000 * 9.81, 13600000.0, 0.0])

    matriz_geometria = np.array([
        [1.0, 1.0, 1.0],      # Suma de fuerzas verticales
        [3.2, 20.1, 20.1],    # Brazos de palanca eje X
        [0.0, -4.5, 4.5]      # Brazos de palanca eje Y
    ])

    # Datos para Regresión Lineal (Telemetría ILS: Distancia a pista vs Altitud)
    x_ils = np.linspace(-10000, -500, 200)
    # Altitud con ruido de medición del radar
    h_ils = 0.0524 * np.abs(x_ils) + 15.0 + np.random.normal(0, 8, 200)

    # Datos para Interpolación 1D (Perfil de viento en la capa límite)
    altitudes_viento = np.array([0, 50, 100, 200, 500, 1000])  # [m]
    velocidad_viento = np.array([2.1, 4.5, 7.2, 8.5, 12.0, 15.3])  # [m/s]

    # Datos para Derivación e Integración (Dinámica de frenado)
    tiempo_frenado = np.linspace(0, 35, 1000)  # [s]
    # Perfil de velocidad [m/s]
    velocidad_frenado = 70.0 * np.exp(-0.04 * tiempo_frenado) - 1.2 * tiempo_frenado
    velocidad_frenado[velocidad_frenado < 0] = 0

    return {
        "params_masa": params_masa,
        "matriz_geometria": matriz_geometria,
        "x_ils": x_ils,
        "h_ils": h_ils,
        "altitudes_viento": altitudes_viento,
        "velocidad_viento": velocidad_viento,
        "tiempo_frenado": tiempo_frenado,
        "velocidad_frenado": velocidad_frenado,
    }


def resolver_reparto_cargas(matriz_geometria, params_masa):
    """Apartado 1: Resuelve A*x = b para hallar [R_m, R_pi, R_pd] en N."""
    peso_total = params_masa[1]
    momento_x = params_masa[2]
    momento_y = params_masa[3]
    b = np.array([peso_total, momento_x, momento_y])

    reacciones = np.linalg.solve(matriz_geometria, b)
    return reacciones


def ajustar_senda_planeo(x_ils, h_ils):
    """Apartado 2: Regresión lineal h = m*x + n por ecuaciones normales."""
    # Matriz de diseño X con columna de unos y columna de x_ils
    X = np.column_stack((np.ones_like(x_ils), x_ils))

    # Ecuaciones normales: theta = (X^T * X)^(-1) * X^T * y
    Xt = X.T
    theta = np.linalg.inv(Xt @ X) @ (Xt @ h_ils)

    n = theta[0]
    m = theta[1]
    h_ajuste = m * x_ils + n

    # Ángulo real de descenso gamma = arctan(m) en grados
    angulo_deg = np.degrees(np.arctan(m))
    return m, n, h_ajuste, angulo_deg


def interpolar_viento_dh(altitudes_viento, velocidad_viento, h_decision=60.0):
    """Apartado 3: Interpolación lineal 1D de la velocidad del viento a h_decision."""
    v_viento_dh = float(np.interp(h_decision, altitudes_viento, velocidad_viento))
    return v_viento_dh


def calcular_t_taxi_newton(v_objetivo=15.0, t0=10.0, tol=1e-8, max_iter=100):
    """Apartado 4: Newton-Raphson para V(t) - v_objetivo = 0 con semilla t0."""
    # f(t) = 70*exp(-0.04*t) - 1.2*t - v_objetivo
    # f'(t) = -2.8*exp(-0.04*t) - 1.2
    t = float(t0)
    iteracion = 0

    while iteracion < max_iter:
        f_t = 70.0 * np.exp(-0.04 * t) - 1.2 * t - v_objetivo
        df_t = -2.8 * np.exp(-0.04 * t) - 1.2

        if abs(f_t) < tol:
            break

        t_siguiente = t - f_t / df_t
        if abs(t_siguiente - t) < tol:
            t = t_siguiente
            break

        t = t_siguiente
        iteracion += 1

    return float(t)


def calcular_aceleracion_frenado(tiempo_frenado, velocidad_frenado):
    """Apartado 5: Aceleración a(t) = dV/dt por diferencias finitas centrales (slicing)."""
    aceleracion = np.zeros_like(velocidad_frenado)

    # Puntos interiores mediante diferencias finitas centrales vectorizadas
    aceleracion[1:-1] = (velocidad_frenado[2:] - velocidad_frenado[:-2]) / (
        tiempo_frenado[2:] - tiempo_frenado[:-2]
    )

    # Extremos con diferencias hacia adelante / atrás de primer orden
    aceleracion[0] = (velocidad_frenado[1] - velocidad_frenado[0]) / (
        tiempo_frenado[1] - tiempo_frenado[0]
    )
    aceleracion[-1] = (velocidad_frenado[-1] - velocidad_frenado[-2]) / (
        tiempo_frenado[-1] - tiempo_frenado[-2]
    )

    return aceleracion


def calcular_distancia_pista(tiempo_frenado, velocidad_frenado):
    """Apartado 6: Distancia S = integral(V(t) dt) por regla del trapecio vectorial."""
    dt = tiempo_frenado[1:] - tiempo_frenado[:-1]
    v_medios = 0.5 * (velocidad_frenado[1:] + velocidad_frenado[:-1])
    distancia_total = float(np.sum(v_medios * dt))
    return distancia_total