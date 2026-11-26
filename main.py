import numpy as np

from calculos_aterrizaje import (
    ajustar_senda_planeo,
    calcular_aceleracion_frenado,
    calcular_distancia_pista,
    calcular_t_taxi_newton,
    cargar_datos,
    interpolar_viento_dh,
    resolver_reparto_cargas,
)
from dashboard_aterrizaje import crear_dashboard


def main():
    np.random.seed(42)
    datos = cargar_datos()

    # 1. Reparto de cargas en el tren de aterrizaje
    reacciones = resolver_reparto_cargas(
        datos["matriz_geometria"], datos["params_masa"]
    )
    print("=== 1. REPARTO DE CARGAS EN EL TREN DE ATERRIZAJE ===")
    print(
        f"R_morro = {reacciones[0]:.2f} N ({reacciones[0]/1000:.2f} kN) | "
        f"R_principal_izq = {reacciones[1]:.2f} N ({reacciones[1]/1000:.2f} kN) | "
        f"R_principal_der = {reacciones[2]:.2f} N ({reacciones[2]/1000:.2f} kN)"
    )
    print(
        "Interpretación física: Los trenes principales soportan la mayor parte del "
        "peso estático y el momento en X, mientras que el tren de morro mantiene el "
        "equilibrio longitudinal sin superar límites estructurales.\n"
    )

    # 2. Regresión lineal de la senda de planeo ILS
    m, n, h_ajuste, angulo_deg = ajustar_senda_planeo(
        datos["x_ils"], datos["h_ils"]
    )
    print("=== 2. CALIBRACIÓN DE LA SENDA DE PLANEO (ILS) ===")
    print(f"Pendiente m = {m:.6f}, Ordenada n = {n:.2f} m, Ángulo gamma = {angulo_deg:.2f}°")
    print(
        f"Interpretación física: Un ángulo de descenso de {angulo_deg:.2f}° se sitúa "
        "en torno al valor nominal de referencia OACI (~3°), lo que permite una "
        "aproximación estable y controlada hacia la zona de toma de contacto.\n"
    )

    # 3. Interpolación 1D del viento en la altitud de decisión (60 m)
    v_viento_dh = interpolar_viento_dh(
        datos["altitudes_viento"], datos["velocidad_viento"], h_decision=60.0
    )
    print("=== 3. VIENTO EN LA ALTURA DE DECISIÓN (DH = 60 m) ===")
    print(f"Velocidad del viento a 60 m = {v_viento_dh:.2f} m/s")
    print(
        "Interpretación física: Al acercarse al suelo, el rozamiento con la "
        "superficie reduce la velocidad del viento respecto a capas superiores, "
        "aunque exige corregir la deriva lateral antes del touchdown.\n"
    )

    # 4. Instante de velocidad de rodaje (V_taxi = 15 m/s) mediante Newton-Raphson
    t_taxi = calcular_t_taxi_newton(v_objetivo=15.0, t0=10.0)
    print("=== 4. INSTANTE DE VELOCIDAD DE RODAJE (NEWTON-RAPHSON) ===")
    print(f"Instante t_taxi (V = 15 m/s) = {t_taxi:.2f} s")
    print(
        "Interpretación física: La aeronave alcanza una velocidad segura de salida "
        f"de pista en torno a los {t_taxi:.1f} s tras iniciar el frenado, permitiendo "
        "despejar la pista de forma eficiente.\n"
    )

    # 5. Perfil de desaceleración por diferencias finitas centrales
    aceleracion = calcular_aceleracion_frenado(
        datos["tiempo_frenado"], datos["velocidad_frenado"]
    )
    max_desaceleracion = float(np.max(np.abs(aceleracion)))
    limite_confort = 0.3 * 9.81
    cumple_confort = max_desaceleracion <= limite_confort
    print("=== 5. PERFIL DE DESACELERACIÓN EN FRENADO ===")
    print(
        f"Máxima desaceleración |a(t)| = {max_desaceleracion:.2f} m/s² "
        f"(Límite 0.3g ≈ {limite_confort:.2f} m/s²)"
    )
    print(
        "Interpretación física: "
        + (
            "La desaceleración máxima no supera el límite de confort de 0.3g, por lo "
            "que el frenado resulta seguro y confortable para los pasajeros."
            if cumple_confort
            else "Se supera el umbral de confort de 0.3g en algún instante del frenado; "
            "conviene revisar la modulación de frenos."
        )
        + "\n"
    )

    # 6. Distancia de pista consumida mediante regla del trapecio vectorial
    distancia = calcular_distancia_pista(
        datos["tiempo_frenado"], datos["velocidad_frenado"]
    )
    print("=== 6. DISTANCIA DE PISTA CONSUMIDA ===")
    print(f"Distancia total recorrida S = {distancia:.2f} m")
    print(
        f"Interpretación física: Se consumen aproximadamente {distancia:.0f} m de "
        "pista hasta detener la aeronave, valor coherente con el recorrido de "
        "frenado de un avión comercial tras el touchdown.\n"
    )

    resultados = {
        "reacciones": reacciones,
        "m": m,
        "n": n,
        "h_ajuste": h_ajuste,
        "angulo_deg": angulo_deg,
        "v_viento_dh": v_viento_dh,
        "t_taxi": t_taxi,
        "aceleracion": aceleracion,
        "distancia": distancia,
    }

    crear_dashboard(datos, resultados)


if __name__ == "__main__":
    main()