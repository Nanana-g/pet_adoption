'''
sistema edos 
Autor: Ariene Víquez

correr despues de nexo_tasas
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

#load
tasas = pd.read_csv('tasas_lambda.csv').set_index('parametro')['valor']

lambda_view     = tasas['lambda_view']
lambda_cart     = tasas['lambda_cart']
lambda_purchase = tasas['lambda_purchase']
lambda_abandon  = tasas['lambda_abandon']
tau             = tasas['tau_horas']

print("~" * 50)
print("  parametros cargados")
print("=" * 50)
print(f"  λ_view     = {lambda_view:.4f}")
print(f"  λ_cart     = {lambda_cart:.4f}")
print(f"  λ_purchase = {lambda_purchase:.4f}")
print(f"  λ_abandon  = {lambda_abandon:.4f}")
print(f"  τ          = {tau:.4f} horas")


df = pd.read_csv('data/pet_adoption_funnel.csv')
N  = len(df)

V0 = N * (1 - df['viewed_product'].mean())          
P0 = df['viewed_product'].sum() - df['added_to_cart'].sum()  
C0 = df['added_to_cart'].sum() - df['purchased'].sum()      
B0 = df['purchased'].sum() 
A0 = df['abandoned'].sum()

print(f"\nCondiciones iniciales:")
print(f"  V0 = {V0:.0f} usuarios")
print(f"  P0 = {P0:.0f} usuarios")
print(f"  C0 = {C0:.0f} usuarios")
print(f"  B0 = {B0:.0f} adopciones")
print(f"  A0 = {A0:.0f} abandonos")

y0 = [V0, P0, C0, B0, A0]

#edos
def funnel_odes(t, y, lv, lc, lp, la, alpha_func):
    """
    Parámetros:
        t: tiempo (horas)
        y: vector de estado [V, P, C, B, A]
        lv: lambda_view — tasa V → P
        lc: lambda_cart — tasa P → C
        lp: lambda_purchase — tasa C → B
        la: lambda_abandon — tasa P → A
        alpha_func: función α(t) — tasa de entrada de nuevos visitantes

    Supuestos documentados:
        - Cada etapa pierde usuarios a tasa proporcional a su población (proceso Markoviano)
        - α(t) puede ser constante o variable (campañas, estacionalidad)
        - B y A son estados absorbentes (no hay retorno)
        - No se modela reingreso al funnel de usuarios que abandonaron
    """

    V, P, C, B, A = y
    alpha = alpha_func(t)

    dV = alpha - lv * V
    dP = lv * V - lc * P - la * P
    dC = lc * P - lp * C
    dB = lp * C
    dA = la * P

    return [dV, dP, dC, dB, dA]

# auxiliar
def resolver_funnel(alpha_func, t_span, t_eval, y0, params, titulo, nombre_archivo):
    lv, lc, lp, la = params
    sol = solve_ivp(
        fun=lambda t, y: funnel_odes(t, y, lv, lc, lp, la, alpha_func),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    V, P, C, B, A = sol.y
    t = sol.t

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(titulo, fontsize=14, fontweight='bold')

    colores = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71', '#e74c3c']
    labels  = ['V(t) Visitantes', 'P(t) Vistas perfil', 'C(t) Solicitudes',
               'B(t) Adopciones', 'A(t) Abandonos']
    datos   = [V, P, C, B, A]

    for i, (ax, data, label, color) in enumerate(zip(axes.flat[:5], datos, labels, colores)):
        ax.plot(t, data, color=color, linewidth=2)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Tiempo (horas)')
        ax.set_ylabel('Usuarios')
        ax.grid(True, alpha=0.3)
        ax.fill_between(t, data, alpha=0.15, color=color)

    
    ax6 = axes.flat[5]
    for data, label, color in zip(datos, labels, colores):
        ax6.plot(t, data, label=label, color=color, linewidth=1.8)
    ax6.set_title('Todas las etapas', fontweight='bold')
    ax6.set_xlabel('Tiempo (horas)')
    ax6.set_ylabel('Usuarios')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'outputs/{nombre_archivo}.png', dpi=150)
    plt.show()

    return sol

params_base = (lambda_view, lambda_cart, lambda_purchase, lambda_abandon)

#escenario 1
#a -> entrada estable de visitantes por hora

T_HORAS= 72          
ALPHA_BASE = N / T_HORAS

t_eval= np.linspace(0, T_HORAS, 1000)

alpha_base= lambda t: ALPHA_BASE

sol1 = resolver_funnel(
    alpha_func= alpha_base,
    t_span = (0, T_HORAS),
    t_eval = t_eval,
    y0= y0,
    params= params_base,
    titulo = 'Escenario 1: Sistema Base — Sin Intervención',
    nombre_archivo= 'escenario1_base'
)

V1, P1, C1, B1, A1 = sol1.y
print(f"\nEscenario 1 — Resultados al final ({T_HORAS}h):")
print(f"  Adopciones totales: {B1[-1]:.0f}")
print(f"  Abandonos totales:  {A1[-1]:.0f}")
print(f"  Tasa conversión:    {B1[-1]/(B1[-1]+A1[-1])*100:.2f}%")

#Escenario 2
def alpha_campana(t):
    if t <= 24:
        return ALPHA_BASE * 3.0   #triple de tráfico durante la campaña
    else:
        return ALPHA_BASE

#campaña mejora λ_cart y λ_purchase un 20%
params_campana = (
    lambda_view,
    lambda_cart* 1.20,
    lambda_purchase * 1.20,
    lambda_abandon * 0.85    # menos abandono con descuento
)

sol2 = resolver_funnel(
    alpha_func = alpha_campana,
    t_span = (0, T_HORAS),
    t_eval = t_eval,
    y0= y0,
    params = params_campana,
    titulo= 'Escenario 2: Campaña de Descuento (primeras 24h)',
    nombre_archivo= 'escenario2_campana'
)

V2, P2, C2, B2, A2 = sol2.y
print(f"\nEscenario 2 — Resultados al final ({T_HORAS}h):")
print(f"  Adopciones totales: {B2[-1]:.0f}")
print(f"  Abandonos totales:  {A2[-1]:.0f}")
print(f"  Tasa conversión:    {B2[-1]/(B2[-1]+A2[-1])*100:.2f}%")
print(f"  Δ adopciones vs base: +{B2[-1]-B1[-1]:.0f} ({(B2[-1]/B1[-1]-1)*100:.1f}%)")

#Escenario 3
# El 55% del tráfico es móvil pero convierte menos.

MOBILE_SHARE = 0.55
MEJORA_UX= 0.35  #35% de mejora en tasas para usuarios móviles

lambda_cart_ux = lambda_cart * (1 + MOBILE_SHARE * MEJORA_UX)
lambda_purchase_ux = lambda_purchase * (1 + MOBILE_SHARE * MEJORA_UX)
lambda_abandon_ux = lambda_abandon * (1 - MOBILE_SHARE * MEJORA_UX * 0.5)

params_ux = (lambda_view, lambda_cart_ux, lambda_purchase_ux, lambda_abandon_ux)

sol3 = resolver_funnel(
    alpha_func = alpha_base,
    t_span= (0, T_HORAS),
    t_eval = t_eval,
    y0= y0,
    params = params_ux,
    titulo = 'Escenario 3: Mejora UX Móvil (+35% conversión mobile)',
    nombre_archivo= 'escenario3_ux_movil'
)

V3, P3, C3, B3, A3 = sol3.y
print(f"\nEscenario 3 — Resultados al final ({T_HORAS}h):")
print(f"  Adopciones totales: {B3[-1]:.0f}")
print(f"  Abandonos totales:  {A3[-1]:.0f}")
print(f"  Tasa conversión:    {B3[-1]/(B3[-1]+A3[-1])*100:.2f}%")
print(f"  Δ adopciones vs base: +{B3[-1]-B1[-1]:.0f} ({(B3[-1]/B1[-1]-1)*100:.1f}%)")


#escenario 4
# Shelter_rating cae → λ_view baja
# y λ_abandon sube (más gente se va sin completar
def alpha_crisis(t):
    if t <= 12:    #tráfico cae 40% después de las primeras 12 horas
        return ALPHA_BASE
    else:
        return ALPHA_BASE * 0.60

params_crisis = (
    lambda_view * 0.70,   # menos gente llega a ver perfiles
    lambda_cart * 0.80,   # menos solicitudes
    lambda_purchase * 0.75,   # menos adopciones completas
    lambda_abandon * 1.50    # más abandono
)

sol4 = resolver_funnel(
    alpha_func = alpha_crisis,
    t_span = (0, T_HORAS),
    t_eval = t_eval,
    y0 = y0,
    params = params_crisis,
    titulo = 'Escenario 4: Crisis de Reputación del Refugio',
    nombre_archivo= 'escenario4_crisis'
)

V4, P4, C4, B4, A4 = sol4.y
print(f"\nEscenario 4 — Resultados al final ({T_HORAS}h):")
print(f"  Adopciones totales: {B4[-1]:.0f}")
print(f"  Abandonos totales:  {A4[-1]:.0f}")
print(f"  Tasa conversión:    {B4[-1]/(B4[-1]+A4[-1])*100:.2f}%")
print(f"  Δ adopciones vs base: {B4[-1]-B1[-1]:.0f} ({(B4[-1]/B1[-1]-1)*100:.1f}%)")

# ── Comparativa final: adopciones B(t) en todos los escenarios
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

escenarios = [
    (sol1.y[3], 'Base',              '#7f8c8d', '-'),
    (sol2.y[3], 'Campaña descuento', '#e67e22', '--'),
    (sol3.y[3], 'Mejora UX móvil',   '#2ecc71', '-.'),
    (sol4.y[3], 'Crisis reputación', '#e74c3c', ':'),
]

# Panel izquierdo: B(t) adopciones acumuladas
for data, label, color, ls in escenarios:
    axes[0].plot(t_eval, data, label=label, color=color, linewidth=2, linestyle=ls)
axes[0].set_title('Adopciones Acumuladas B(t)', fontweight='bold')
axes[0].set_xlabel('Tiempo (horas)')
axes[0].set_ylabel('Adopciones')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel derecho: A(t) abandonos acumulados
abandonos = [
    (sol1.y[4], 'Base',              '#7f8c8d', '-'),
    (sol2.y[4], 'Campaña descuento', '#e67e22', '--'),
    (sol3.y[4], 'Mejora UX móvil',   '#2ecc71', '-.'),
    (sol4.y[4], 'Crisis reputación', '#e74c3c', ':'),
]
for data, label, color, ls in abandonos:
    axes[1].plot(t_eval, data, label=label, color=color, linewidth=2, linestyle=ls)
axes[1].set_title('Abandonos Acumulados A(t)', fontweight='bold')
axes[1].set_xlabel('Tiempo (horas)')
axes[1].set_ylabel('Abandonos')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Comparativa de Escenarios — Sistema de EDOs', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/comparativa_escenarios.png', dpi=150)
plt.show()

#comparaciones de escenarios
resumen = pd.DataFrame({
    'Escenario': ['Base', 'Campaña descuento', 'Mejora UX móvil', 'Crisis reputación'],
    'Adopciones':[int(sol1.y[3][-1]), int(sol2.y[3][-1]), int(sol3.y[3][-1]), int(sol4.y[3][-1])],
    'Abandonos': [int(sol1.y[4][-1]), int(sol2.y[4][-1]), int(sol3.y[4][-1]), int(sol4.y[4][-1])],
    'Conv_%': [
        round(sol1.y[3][-1]/(sol1.y[3][-1]+sol1.y[4][-1])*100, 2),
        round(sol2.y[3][-1]/(sol2.y[3][-1]+sol2.y[4][-1])*100, 2),
        round(sol3.y[3][-1]/(sol3.y[3][-1]+sol3.y[4][-1])*100, 2),
        round(sol4.y[3][-1]/(sol4.y[3][-1]+sol4.y[4][-1])*100, 2),
    ]
})
print("\n" + "~" * 55)
print(" Tabla comparativa de escenarios")
print(",+" * 27)
print(resumen.to_string(index=False))
resumen.to_csv('outputs/resumen_escenarios.csv', index=False)
print("\nTodos los resultados guardados en outputs/")

