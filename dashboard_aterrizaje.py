import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np

from calculos_aterrizaje import (
    calcular_aceleracion_frenado,
    calcular_distancia_pista,
    calcular_t_taxi_newton,
    interpolar_viento_dh,
)


def crear_dashboard(datos, resultados):
    """Crea la única ventana interactiva con RadioButtons, Slider y anotaciones."""
    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title(
        "Propuesta 4 - Senda de Aterrizaje y Dinámica de Frenado"
    )

    # Panel lateral izquierdo para RadioButtons
    ax_radio = plt.axes([0.03, 0.58, 0.22, 0.28])
    radio = RadioButtons(ax_radio, ("Aproximación", "Atmósfera", "Frenado"))

    # Eje superior derecho para texto numérico dinámico
    ax_texto = plt.axes([0.30, 0.80, 0.66, 0.14])
    ax_texto.axis("off")

    # Área principal de gráficos (se reconfigura según la vista activa)
    ax_principal = plt.axes([0.32, 0.20, 0.64, 0.54])
    ax_secundario = plt.axes([0.32, 0.12, 0.64, 0.26])
    ax_secundario.set_visible(False)

    # Slider interactivo en la parte inferior (para V_taxi en la vista Frenado)
    ax_slider = plt.axes([0.38, 0.03, 0.50, 0.035])
    slider_v_taxi = Slider(
        ax=ax_slider,
        label="V_taxi [m/s]",
        valmin=5.0,
        valmax=30.0,
        valinit=15.0,
        valstep=0.5,
    )
    ax_slider.set_visible(False)

    estado = {"vista": "Aproximación", "v_taxi": 15.0}

    def dibujar():
        vista = estado["vista"]
        v_taxi_actual = estado["v_taxi"]

        ax_principal.clear()
        ax_secundario.clear()
        ax_texto.clear()
        ax_texto.axis("off")

        if vista == "Aproximación":
            ax_secundario.set_visible(False)
            ax_slider.set_visible(False)
            ax_principal.set_position([0.32, 0.15, 0.64, 0.60])

            x_ils = datos["x_ils"]
            h_ils = datos["h_ils"]
            h_ajuste = resultados["h_ajuste"]
            m = resultados["m"]
            n = resultados["n"]
            angulo_deg = resultados["angulo_deg"]
            reacciones = resultados["reacciones"]

            ax_principal.scatter(
                x_ils, h_ils, s=14, alpha=0.65, label="Telemetría ILS (radar)"
            )
            ax_principal.plot(
                x_ils,
                h_ajuste,
                linewidth=2.2,
                label=f"Ajuste lineal: h = {m:.5f}·x + {n:.2f} (ángulo: {angulo_deg:.2f}°)",
            )
            ax_principal.set_title("Aproximación ILS y Senda de Planeo")
            ax_principal.set_xlabel("Distancia horizontal a pista x [m]")
            ax_principal.set_ylabel("Altitud h [m]")
            ax_principal.grid(True, linestyle="--", alpha=0.5)
            ax_principal.legend(loc="upper left")

            texto_res = (
                f"Cargas en apoyos -> Morro (R_m): {reacciones[0]/1000:.2f} kN | "
                f"P. Izq (R_pi): {reacciones[1]/1000:.2f} kN | "
                f"P. Der (R_pd): {reacciones[2]/1000:.2f} kN\n"
                f"Ángulo de descenso real γ: {angulo_deg:.2f}° (Referencia OACI ~3.0°)"
            )
            ax_texto.text(
                0.0, 0.5, texto_res, fontsize=9.5, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f6f9", edgecolor="#888")
            )

        elif vista == "Atmósfera":
            ax_secundario.set_visible(False)
            ax_slider.set_visible(False)
            ax_principal.set_position([0.32, 0.15, 0.64, 0.60])

            alt_v = datos["altitudes_viento"]
            vel_v = datos["velocidad_viento"]
            v_dh = interpolar_viento_dh(alt_v, vel_v, h_decision=60.0)

            ax_principal.plot(
                vel_v, alt_v, marker="o", linewidth=2.0, label="Perfil de viento medido"
            )
            ax_principal.scatter(
                [v_dh], [60.0], color="red", s=70, zorder=5,
                label=f"Altitud de decisión (60 m): {v_dh:.2f} m/s"
            )
            ax_principal.axhline(60.0, color="red", linestyle=":", alpha=0.6)
            ax_principal.set_title("Perfil de Viento en la Capa Límite y Altitud de Decisión")
            ax_principal.set_xlabel("Velocidad del viento [m/s]")
            ax_principal.set_ylabel("Altitud [m]")
            ax_principal.grid(True, linestyle="--", alpha=0.5)
            ax_principal.legend(loc="lower right")

            texto_res = (
                f"Velocidad del viento interpolada a DH = 60 m: {v_dh:.2f} m/s "
                f"({v_dh * 3.6:.1f} km/h)"
            )
            ax_texto.text(
                0.0, 0.5, texto_res, fontsize=9.5, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f6f9", edgecolor="#888")
            )

        elif vista == "Frenado":
            ax_secundario.set_visible(True)
            ax_slider.set_visible(True)

            ax_principal.set_position([0.32, 0.48, 0.64, 0.27])
            ax_secundario.set_position([0.32, 0.12, 0.64, 0.25])

            t_f = datos["tiempo_frenado"]
            v_f = datos["velocidad_frenado"]
            a_f = resultados["aceleracion"]
            distancia = resultados["distancia"]
            t_taxi = calcular_t_taxi_newton(v_objetivo=v_taxi_actual, t0=10.0)
            v_en_taxi = 70.0 * np.exp(-0.04 * t_taxi) - 1.2 * t_taxi

            # Subgráfica superior: Velocidad y área integrada (distancia)
            ax_principal.plot(t_f, v_f, linewidth=2.0, label="Velocidad de frenado V(t)")
            ax_principal.fill_between(
                t_f, 0, v_f, alpha=0.25, label=f"Distancia consumida S = {distancia:.1f} m"
            )
            ax_principal.scatter(
                [t_taxi], [v_en_taxi], color="red", s=55, zorder=5,
                label=f"t_taxi ({v_taxi_actual:.1f} m/s) = {t_taxi:.2f} s"
            )
            ax_principal.set_title("Dinámica de Frenado: Velocidad y Aceleración")
            ax_principal.set_ylabel("Velocidad [m/s]")
            ax_principal.grid(True, linestyle="--", alpha=0.5)
            ax_principal.legend(loc="upper right", fontsize=8.5)

            # Subgráfica inferior: Aceleración y límite 0.3g (~2.94 m/s^2)
            ax_secundario.plot(
                t_f, a_f, color="tab:orange", linewidth=1.8,
                label="Aceleración a(t) [diferencias finitas centrales]"
            )
            ax_secundario.axhline(
                -2.94, color="red", linestyle="--", linewidth=1.3,
                label="Límite confort (-0.3g ≈ -2.94 m/s²)"
            )
            ax_secundario.set_xlabel("Tiempo de frenado [s]")
            ax_secundario.set_ylabel("Aceleración [m/s²]")
            ax_secundario.grid(True, linestyle="--", alpha=0.5)
            ax_secundario.legend(loc="lower right", fontsize=8.5)

            max_desac = float(np.max(np.abs(a_f)))
            texto_res = (
                f"t_taxi (V = {v_taxi_actual:.1f} m/s): {t_taxi:.2f} s | "
                f"Distancia total S: {distancia:.1f} m | "
                f"Máx. |a(t)|: {max_desac:.2f} m/s² (Límite 0.3g ≈ 2.94 m/s²)"
            )
            ax_texto.text(
                0.0, 0.5, texto_res, fontsize=9.2, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f6f9", edgecolor="#888")
            )

        fig.canvas.draw_idle()

    def al_cambiar_vista(etiqueta):
        estado["vista"] = etiqueta
        dibujar()

    def al_mover_slider(valor):
        estado["v_taxi"] = float(valor)
        dibujar()

    radio.on_clicked(al_cambiar_vista)
    slider_v_taxi.on_changed(al_mover_slider)

    dibujar()
    plt.show()