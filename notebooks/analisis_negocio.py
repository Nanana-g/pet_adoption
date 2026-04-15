'''
analisis de negocio
autor Ariana Víquez

este codigo se debe correr al tener todos los outputs para el trabajo

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from itertools import product
import warnings
warnings.filterwarnings('ignore')

#carga de datos
df = pd.read_csv('data/pet_adoption_funnel.csv')
tasas = pd.read_csv('outputs/tasas_lambda.csv').set_index('parametro')['valor']

LAMBDA_VIEW = float(tasas['lambda_view'])
LAMBDA_CART = float(tasas['lambda_cart'])
LAMBDA_PURCHASE = float(tasas['lambda_purchase'])
LAMBDA_ABANDON = float(tasas['lambda_abandon'])
N = len(df)
T_HORAS = 72
ALPHA_BASE = N / T_HORAS

#reviciones del sistema
def funnel_odes(t, y, lv, lc, lp, la, alpha_func):
    V, P, C, B, A = y
    dV = alpha_func(t) - lv * V
    dP = lv * V - lc * P - la * P
    dC = lc * P - lp * C
    dB = lp * C
    dA = la * P
    return [dV, dP, dC, dB, dA]

def resolver(lv, lc, lp, la, alpha_func, t_horas=T_HORAS, puntos=500):
    V0 = N * (1 - df['viewed_product'].mean())
    P0 = df['viewed_product'].sum() - df['added_to_cart'].sum()
    C0 = df['added_to_cart'].sum()  - df['purchased'].sum()
    B0 = float(df['purchased'].sum())
    A0 = float(df['abandoned'].sum())
    y0 = [V0, P0, C0, B0, A0]
    t_eval = np.linspace(0, t_horas, puntos)
    sol = solve_ivp(
        fun = lambda t, y: funnel_odes(t, y, lv, lc, lp, la, alpha_func),
        t_span = (0, t_horas),
        y0 = y0,
        t_eval = t_eval,
        method = 'RK45', rtol=1e-6, atol=1e-8
    )
    return sol.t, sol.y

print("Datos y modelos cargados correctamente")

#p. 1: análisis de sensibilidad de tasas
# Variamos cada λ ±50% mientras las demás se mantienen fijas y medimos el impacto en adopciones finales B(T)

VARIACIONES = np.linspace(0.5, 1.5, 20)   # de -50% a +50%
alpha_cte = lambda t: ALPHA_BASE

resultados_sens = {
    'lambda_view': [],
    'lambda_cart': [],
    'lambda_purchase': [],
    'lambda_abandon': [],
}

for mult in VARIACIONES: #variar lambda_view
    _, y = resolver(LAMBDA_VIEW*mult, LAMBDA_CART, LAMBDA_PURCHASE, LAMBDA_ABANDON, alpha_cte)
    resultados_sens['lambda_view'].append(y[3][-1])

    _, y = resolver(LAMBDA_VIEW, LAMBDA_CART*mult, LAMBDA_PURCHASE, LAMBDA_ABANDON, alpha_cte)
    resultados_sens['lambda_cart'].append(y[3][-1])#lambda_cart

    _, y = resolver(LAMBDA_VIEW, LAMBDA_CART, LAMBDA_PURCHASE*mult, LAMBDA_ABANDON, alpha_cte)
    resultados_sens['lambda_purchase'].append(y[3][-1]) #lambda_purchase

    _, y = resolver(LAMBDA_VIEW, LAMBDA_CART, LAMBDA_PURCHASE, LAMBDA_ABANDON*mult, alpha_cte)
    resultados_sens['lambda_abandon'].append(y[3][-1])#lambda_abandon

_, y_base = resolver(LAMBDA_VIEW, LAMBDA_CART, LAMBDA_PURCHASE, LAMBDA_ABANDON, alpha_cte)
B_base = y_base[3][-1] #elasticidad: % cambio en B / % cambio en λ

elasticidades = {}
for param, valores in resultados_sens.items():
    delta_B = (valores[-1] - valores[0]) / B_base
    delta_L = (VARIACIONES[-1] - VARIACIONES[0])
    elasticidades[param] = round(delta_B / delta_L, 4)

print("\n" + "~" * 55)
print("  p. 1: ¿Qué tasa impacta más las adopciones?")
print("~" * 55)
print("\nElasticidad de B(T) respecto a cada λ:")
for k, v in sorted(elasticidades.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:20s} → elasticidad = {v:+.4f}")

# Gráfica
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colores = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
params  = ['lambda_view', 'lambda_cart', 'lambda_purchase', 'lambda_abandon']
nombres = ['λ_view (V→P)', 'λ_cart (P→C)', 'λ_purchase (C→B)', 'λ_abandon (P→A)']

for ax, param, nombre, color in zip(axes.flat, params, nombres, colores):
    ax.plot(VARIACIONES * 100 - 100, resultados_sens[param],
            color=color, linewidth=2.5)
    ax.axhline(y=B_base, color='gray', linestyle='--', linewidth=1, label='Base')
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
    ax.set_title(f'Sensibilidad: {nombre}', fontweight='bold')
    ax.set_xlabel('Variación del parámetro (%)')
    ax.set_ylabel('Adopciones finales B(T)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    e = elasticidades[param]
    ax.text(0.05, 0.92, f'Elasticidad: {e:+.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

plt.suptitle('Análisis de Sensibilidad — Impacto de cada λ en Adopciones',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/p1_sensibilidad_tasas.png', dpi=150)
plt.show()

print(f"\n Respuesta: La tasa con mayor elasticidad es la que más conviene optimizar primero")



#p. 2: Timing óptimo de campaña 
#simulacion del mismo spike de tráfico (3x) pero iniciándolo en distintos momentos del horizonte de 72 horas

TIEMPOS_INICIO = [0, 6, 12, 24, 36, 48]
DURACION_CAMPANA = 12   #horas del spike
resultados_timing = []

for t_inicio in TIEMPOS_INICIO:
    def alpha_timing(t, ti=t_inicio):
        if ti <= t <= ti + DURACION_CAMPANA:
            return ALPHA_BASE * 3.0
        return ALPHA_BASE

    _, y = resolver(LAMBDA_VIEW, LAMBDA_CART, LAMBDA_PURCHASE,
                    LAMBDA_ABANDON, alpha_timing)
    resultados_timing.append({
        't_inicio': t_inicio,
        'adopciones': round(y[3][-1], 1),
        'abandonos': round(y[4][-1], 1),
        'conv_rate': round(y[3][-1] / (y[3][-1] + y[4][-1]) * 100, 2)
    })

df_timing = pd.DataFrame(resultados_timing)
mejor = df_timing.loc[df_timing['adopciones'].idxmax()]

print("\n" + "~" * 55)
print("  p. 2: Cuándo lanzar la campaña?")
print("~" * 55)
print(df_timing.to_string(index=False))
print(f"\n mejor momento: hora {int(mejor['t_inicio'])} -> {mejor['adopciones']:.0f} adopciones")

# Gráfica
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colores_t = plt.cm.plasma(np.linspace(0.1, 0.9, len(TIEMPOS_INICIO)))

t_eval = np.linspace(0, T_HORAS, 500)
for i, t_inicio in enumerate(TIEMPOS_INICIO):
    def alpha_timing(t, ti=t_inicio):
        return ALPHA_BASE * 3.0 if ti <= t <= ti + DURACION_CAMPANA else ALPHA_BASE
    t_sol, y = resolver(LAMBDA_VIEW, LAMBDA_CART, LAMBDA_PURCHASE,
                        LAMBDA_ABANDON, alpha_timing)
    axes[0].plot(t_sol, y[3], color=colores_t[i], linewidth=2,
                 label=f'Inicio h={t_inicio}')

axes[0].set_title('B(t) Adopciones según timing de campaña', fontweight='bold')
axes[0].set_xlabel('Tiempo (horas)')
axes[0].set_ylabel('Adopciones acumuladas')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].bar(df_timing['t_inicio'].astype(str).apply(lambda x: f'h={x}'),
            df_timing['adopciones'], color=colores_t, edgecolor='black', linewidth=0.7)
axes[1].set_title('Adopciones finales por hora de inicio', fontweight='bold')
axes[1].set_xlabel('Hora de inicio de campaña')
axes[1].set_ylabel('Adopciones totales B(T)')
axes[1].grid(True, alpha=0.3, axis='y')
for i, row in df_timing.iterrows():
    axes[1].text(i, row['adopciones'] + 5, f"{row['adopciones']:.0f}",
                 ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Análisis de Timing — Cuándo lanzar la campaña?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/p2_timing_campana.png', dpi=150)
plt.show()


#p. 3: Mapa de calor y combinaciones de mejora 
#Simulamos mejoras simultáneas en pares de tasas y medimos el resultado en adopciones finales

MEJORA = np.linspace(1.0, 1.5, 10)   # de 0% a +50% de mejora

#par 1: lambda_cart vs lambda_purchase (las más accionables)
grid_B = np.zeros((len(MEJORA), len(MEJORA)))

for i, m_cart in enumerate(MEJORA):
    for j, m_purchase in enumerate(MEJORA):
        _, y = resolver(
            LAMBDA_VIEW,
            LAMBDA_CART * m_cart,
            LAMBDA_PURCHASE * m_purchase,
            LAMBDA_ABANDON,
            lambda t: ALPHA_BASE
        )
        grid_B[i, j] = y[3][-1]


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(grid_B, origin='lower', aspect='auto',
                     cmap='YlOrRd',
                     extent=[0, 50, 0, 50])
plt.colorbar(im1, ax=axes[0], label='Adopciones finales B(T)')
axes[0].set_xlabel('Mejora en λ_purchase (%)')
axes[0].set_ylabel('Mejora en λ_cart (%)')
axes[0].set_title('Mapa de Calor: λ_cart × λ_purchase', fontweight='bold')

 
idx_max = np.unravel_index(np.argmax(grid_B), grid_B.shape)
axes[0].plot(idx_max[1] * 50/9, idx_max[0] * 50/9,
             'b*', markersize=15, label='Óptimo')
axes[0].legend()

#par 2: lambda_view vs lambda_abandon
grid_B2 = np.zeros((len(MEJORA), len(MEJORA)))
REDUCCION = np.linspace(1.0, 0.5, 10)   # reducir abandono de 0% a -50%

for i, m_view in enumerate(MEJORA):
    for j, r_abandon in enumerate(REDUCCION):
        _, y = resolver(
            LAMBDA_VIEW * m_view,
            LAMBDA_CART,
            LAMBDA_PURCHASE,
            LAMBDA_ABANDON  * r_abandon,
            lambda t: ALPHA_BASE
        )
        grid_B2[i, j] = y[3][-1]


im2 = axes[1].imshow(grid_B2, origin='lower', aspect='auto',
                     cmap='YlOrRd',
                     extent=[0, 50, 0, 50])
plt.colorbar(im2, ax=axes[1], label='Adopciones finales B(T)')
axes[1].set_xlabel('Reducción en λ_abandon (%)')
axes[1].set_ylabel('Mejora en λ_view (%)')
axes[1].set_title('Mapa de Calor: λ_view × λ_abandon', fontweight='bold')

idx_max2 = np.unravel_index(np.argmax(grid_B2), grid_B2.shape)
axes[1].plot(idx_max2[1] * 50/9, idx_max2[0] * 50/9,
             'b*', markersize=15, label='Óptimo')
axes[1].legend()

plt.suptitle('p. 3:Qué combinación de mejoras maximiza adopciones?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/p3_mapa_calor_combinaciones.png', dpi=150)
plt.show()

print("\n" + "~" * 55)
print("  p. 3: Combinación óptima de mejoras")
print("~" * 55)
print(f"  Mejor mejora λ_cart:     +{(MEJORA[idx_max[0]]-1)*100:.0f}%")
print(f"  Mejor mejora λ_purchase: +{(MEJORA[idx_max[1]]-1)*100:.0f}%")
print(f"  Adopciones resultantes:   {grid_B[idx_max]:.0f}")
print(f"  Δ vs base:               +{grid_B[idx_max]-B_base:.0f} adopciones")



#p. 4: Tiempo de recuperación post-crisis
# simul de crisis de 24h y luego recuperación gradual cuándo B(t) vuelve a cruzar el nivel base

DURACION_CRISIS = 24   # horas de crisis
def alpha_recuperacion(t):
    if t <= DURACION_CRISIS:
        return ALPHA_BASE * 0.40   # crisis: 60% menos tráfico
    else:
        # vuelve al 100% en 24h adicionales
        progreso = min((t - DURACION_CRISIS) / 24, 1.0)
        return ALPHA_BASE * (0.40 + 0.60 * progreso)

params_crisis = (
    LAMBDA_VIEW * 0.70,
    LAMBDA_CART * 0.80,
    LAMBDA_PURCHASE * 0.75,
    LAMBDA_ABANDON * 1.50
)

# 120 horas para ver recuperación completa
T_EXT = 120
t_crisis, y_crisis = resolver(*params_crisis, alpha_recuperacion, t_horas=T_EXT, puntos=600)
t_base,   y_base_ext = resolver(LAMBDA_VIEW, LAMBDA_CART, LAMBDA_PURCHASE,
                                 LAMBDA_ABANDON, lambda t: ALPHA_BASE,
                                 t_horas=T_EXT, puntos=600)

#recuperación
diff = y_base_ext[3] - y_crisis[3]
cruces = np.where(np.diff(np.sign(diff)))[0]
t_recuperacion = t_crisis[cruces[0]] if len(cruces) > 0 else None

print("\n" + "=" * 55)
print("  p. 4: Cuánto tarda en recuperarse el funnel?")
print("=" * 55)
if t_recuperacion:
    print(f"  Tiempo de recuperación: {t_recuperacion:.1f} horas ({t_recuperacion/24:.1f} días)")
    print(f"  Adopciones perdidas durante crisis: {y_base_ext[3][int(DURACION_CRISIS/T_EXT*600)] - y_crisis[3][int(DURACION_CRISIS/T_EXT*600)]:.0f}")
else:
    print(" El funnel NO se recupera dentro del horizonte simulado.")



fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(t_base,   y_base_ext[3], color='#2ecc71', linewidth=2,
             label='Escenario base', linestyle='--')
axes[0].plot(t_crisis, y_crisis[3],   color='#e74c3c', linewidth=2,
             label='Post-crisis')
axes[0].axvline(x=DURACION_CRISIS, color='gray', linestyle=':', linewidth=1.5,
                label=f'Fin crisis (h={DURACION_CRISIS})')
if t_recuperacion:
    axes[0].axvline(x=t_recuperacion, color='#3498db', linestyle='-.', linewidth=1.5,
                    label=f'Recuperación (h={t_recuperacion:.0f})')
axes[0].fill_between(t_crisis, y_crisis[3], y_base_ext[3],
                     where=(y_crisis[3] < y_base_ext[3]),
                     alpha=0.2, color='red', label='Adopciones perdidas')
axes[0].set_title('B(t): Recuperación post-crisis', fontweight='bold')
axes[0].set_xlabel('Tiempo (horas)')
axes[0].set_ylabel('Adopciones acumuladas')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)


# Tráfico α(t) durante la recuperación
alpha_vals = [alpha_recuperacion(t) for t in t_crisis]
axes[1].plot(t_crisis, alpha_vals, color='#9b59b6', linewidth=2)
axes[1].axhline(y=ALPHA_BASE, color='gray', linestyle='--', linewidth=1, label='α base')
axes[1].axvline(x=DURACION_CRISIS, color='gray', linestyle=':', linewidth=1.5)
axes[1].set_title('α(t): Tráfico durante crisis y recuperación', fontweight='bold')
axes[1].set_xlabel('Tiempo (horas)')
axes[1].set_ylabel('Visitantes/hora')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('p. 4: Tiempo de recuperación post-crisis',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/p4_recuperacion_crisis.png', dpi=150)
plt.show()




#p. 5: Perfil óptimo de usuario 
#load probabilidades individuales y construimos el perfil del usuario con mayor p_full_funnel

df_prob = pd.read_csv('outputs/probabilidades_individuales.csv')
df_full = df.merge(df_prob[['user_id', 'p_view', 'p_cart', 'p_purchase']], on='user_id')
df_full['p_full_funnel'] = df_full['p_view'] * df_full['p_cart'].fillna(0) * df_full['p_purchase'].fillna(0)

# Top 10% de usuarios con mayor probabilidad de completar el funnel
top_10 = df_full[df_full['p_full_funnel'] >= df_full['p_full_funnel'].quantile(0.90)]
bot_10 = df_full[df_full['p_full_funnel'] <= df_full['p_full_funnel'].quantile(0.10)]


print("\n" + "~" * 55)
print("  p. 5: Perfil del usuario con mayor conversión")
print("~" * 55)

variables_perfil = ['species', 'source', 'device', 'campaign_type', 'weekday']
for var in variables_perfil:
    top_mode = top_10[var].mode()[0]
    bot_mode = bot_10[var].mode()[0]
    print(f"  {var:20s} → Top 10%: {top_mode:15s} | Bot 10%: {bot_mode}")

print(f"\n  session_time promedio -> Top 10%: {top_10['session_time'].mean():.2f} min | Bot 10%: {bot_10['session_time'].mean():.2f} min")
print(f"  returning_user -> Top 10%: {top_10['returning_user'].mean():.2%} | Bot 10%: {bot_10['returning_user'].mean():.2%}")
print(f"  discount_seen -> Top 10%: {top_10['discount_seen'].mean():.2%} | Bot 10%: {bot_10['discount_seen'].mean():.2%}")


#distribución de p_full_funnel por especie y device
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

order_esp = df_full.groupby('species')['p_full_funnel'].median().sort_values(ascending=False).index
df_full.boxplot(column='p_full_funnel', by='species', ax=axes[0],
                positions=range(len(order_esp)),
                patch_artist=True)
axes[0].set_xticklabels(order_esp, rotation=15)
axes[0].set_title('P(full funnel) por especie', fontweight='bold')
axes[0].set_xlabel('Especie')
axes[0].set_ylabel('P(completar funnel)')
axes[0].grid(True, alpha=0.3, axis='y')
plt.sca(axes[0])
plt.title('P(full funnel) por especie')

order_dev = df_full.groupby('device')['p_full_funnel'].median().sort_values(ascending=False).index
df_full.boxplot(column='p_full_funnel', by='device', ax=axes[1],
                positions=range(len(order_dev)),
                patch_artist=True)
axes[1].set_xticklabels(order_dev, rotation=15)
axes[1].set_title('P(full funnel) por dispositivo', fontweight='bold')
axes[1].set_xlabel('Dispositivo')
axes[1].set_ylabel('P(completar funnel)')
axes[1].grid(True, alpha=0.3, axis='y')
plt.sca(axes[1])
plt.title('P(full funnel) por dispositivo')

plt.suptitle('p. 5: Qué perfil de usuario convierte más?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/p5_perfil_usuario_optimo.png', dpi=150)
plt.show()





#------------------------------------------------------------------------
#resumen de decisiones
print("\n" + "~" * 65)
print("  Resumen de decisiones de negocio")
print("=" * 65)

decisiones = pd.DataFrame({
    'Pregunta': [
        'Qué tasa optimizar primero?',
        'Cuándo lanzar la campaña?',
        'Qué mejoras combinar?',
        'Cuánto dura el impacto de una crisis?',
        'A qué perfil de usuario priorizar?'
    ],
    'Hallazgo': [
        f"λ con mayor elasticidad: {max(elasticidades, key=lambda k: abs(elasticidades[k]))}",
        f"Hora óptima de inicio: h={int(mejor['t_inicio'])} ({mejor['adopciones']:.0f} adopciones)",
        f"λ_cart +{(MEJORA[idx_max[0]]-1)*100:.0f}% + λ_purchase +{(MEJORA[idx_max[1]]-1)*100:.0f}%",
        f"Recuperación en {t_recuperacion:.0f}h" if t_recuperacion else "No se recupera en 120h",
        f"Especie: {top_10['species'].mode()[0]}, Device: {top_10['device'].mode()[0]}, Source: {top_10['source'].mode()[0]}"
    ],
    'Acción recomendada': [
        'Invertir en mejorar esa etapa del funnel primero',
        'Programar campañas en ese horario para maximizar ROI',
        'Asignar presupuesto a UX de solicitud y proceso de adopción',
        'Tener plan de contingencia activo antes de que ocurra',
        'Segmentar campañas hacia ese perfil específico'
    ]
})

print(decisiones.to_string(index=False))
decisiones.to_csv('outputs/resumen_decisiones_negocio.csv', index=False)
print("\n Análisis guardado en outputs/")

