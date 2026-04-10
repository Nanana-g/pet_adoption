'''
Nexo tasas 
Autor: Ariene Víquez

correr despues de naive bayes
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#load dataset original y prob del NB
df_raw  = pd.read_csv('data/pet_adoption_funnel.csv')
df_prob = pd.read_csv('outputs/probabilidades_individuales.csv')

df = df_raw.merge(df_prob[['user_id', 'p_view', 'p_cart', 'p_purchase']], on='user_id')

print(df[['user_id', 'p_view', 'p_cart', 'p_purchase']].describe())

#tasas
N_total   = len(df)
N_view    = df['viewed_product'].sum()
N_cart    = df['added_to_cart'].sum()
N_purchase= df['purchased'].sum()
N_abandon = df['abandoned'].sum()

p_bruta_view     = N_view / N_total
p_bruta_cart     = N_cart / N_view
p_bruta_purchase = N_purchase / N_cart
p_bruta_abandon  = N_abandon / N_view   # abandono sobre quienes vieron

print("~" * 50)
print("  Tasa brutas empíricas")
print("=" * 50)
print(f"  P(vista | visita)       = {p_bruta_view:.4f}")
print(f"  P(carrito | vista)      = {p_bruta_cart:.4f}")
print(f"  P(compra | carrito)     = {p_bruta_purchase:.4f}")
print(f"  P(abandono | vista)     = {p_bruta_abandon:.4f}")

#pob promedio de naive bayes
p_nb_view     = df['p_view'].mean()
p_nb_cart     = df.loc[df['viewed_product'] == 1, 'p_cart'].mean()
p_nb_purchase = df.loc[df['added_to_cart'] == 1, 'p_purchase'].mean()

print("\n" + "=" * 50)
print("  Probabiliadades promedio de Naive Bayes")
print("=" * 50)
print(f"  p̄(vista)     = {p_nb_view:.4f}")
print(f"  p̄(carrito)   = {p_nb_cart:.4f}")
print(f"  p̄(compra)    = {p_nb_purchase:.4f}")

#Tabla
comparacion = pd.DataFrame({
    'Transición':   ['Visita → Vista', 'Vista → Carrito', 'Carrito → Compra'],
    'P_empirica':   [p_bruta_view, p_bruta_cart, p_bruta_purchase],
    'P_NaiveBayes': [p_nb_view, p_nb_cart, p_nb_purchase],
})
comparacion['Diferencia_%'] = abs(
    comparacion['P_empirica'] - comparacion['P_NaiveBayes']
) * 100

print("\n")
print(comparacion.to_string(index=False))
comparacion.to_csv('outputs/comparacion_probabilidades.csv', index=False)

tau_horas = df['session_time'].mean() / 60   # convertir minutos a horas
print(f"\nτ promedio de sesión = {tau_horas:.4f} horas ({df['session_time'].mean():.2f} min)")

def prob_a_tasa(p, tau):
    """
    Convierte probabilidad de transición p en tasa λ
    bajo supuesto de distribución exponencial.
    λ = -ln(1 - p) / τ
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)   # evitar log(0)
    return -np.log(1 - p) / tau

lambda_view     = prob_a_tasa(p_nb_view,     tau_horas)
lambda_cart     = prob_a_tasa(p_nb_cart,     tau_horas)
lambda_purchase = prob_a_tasa(p_nb_purchase, tau_horas)
lambda_abandon  = prob_a_tasa(p_bruta_abandon, tau_horas)

print("\n" + "~" * 50)
print("  tasas λ deriadas ")
print("~" * 50)
print(f"  λ_view     (V→P) = {lambda_view:.4f} usuarios/hora")
print(f"  λ_cart     (P→C) = {lambda_cart:.4f} usuarios/hora")
print(f"  λ_purchase (C→B) = {lambda_purchase:.4f} usuarios/hora")
print(f"  λ_abandon  (P→A) = {lambda_abandon:.4f} usuarios/hora")

# Guardar tasas para el Día 4
tasas = pd.DataFrame({
    'parametro': ['lambda_view', 'lambda_cart', 'lambda_purchase', 'lambda_abandon', 'tau_horas'],
    'valor':     [lambda_view, lambda_cart, lambda_purchase, lambda_abandon, tau_horas],
    'descripcion': [
        'Tasa V→P: visita a vista de perfil',
        'Tasa P→C: vista a solicitud de adopción',
        'Tasa C→B: solicitud a adopción completa',
        'Tasa P→A: vista a abandono',
        'Tiempo medio de sesión en horas'
    ]
})
tasas.to_csv('tasas_lambda.csv', index=False)
print("\nTasas guardadas en tasas_lambda.csv")


segmentos = []

for especie in df['species'].unique():
    sub = df[df['species'] == especie]
    
    p_v = sub['p_view'].mean()
    p_c = sub.loc[sub['viewed_product'] == 1, 'p_cart'].mean()
    p_p = sub.loc[sub['added_to_cart'] == 1, 'p_purchase'].mean()
    tau = sub['session_time'].mean() / 60
    
    if pd.isna(p_c): p_c = p_nb_cart
    if pd.isna(p_p): p_p = p_nb_purchase

    segmentos.append({
        'segmento':        especie,
        'n_usuarios':      len(sub),
        'tau_horas':       round(tau, 4),
        'p_view':          round(p_v, 4),
        'p_cart':          round(p_c, 4),
        'p_purchase':      round(p_p, 4),
        'lambda_view':     round(prob_a_tasa(p_v, tau), 4),
        'lambda_cart':     round(prob_a_tasa(p_c, tau), 4),
        'lambda_purchase': round(prob_a_tasa(p_p, tau), 4),
    })

df_segmentos = pd.DataFrame(segmentos).sort_values('lambda_purchase', ascending=False)
print("\nTasas λ por especie:")
print(df_segmentos.to_string(index=False))
df_segmentos.to_csv('outputspor_especie.csv', index=False)

#grafs
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(comparacion))
width = 0.35
bars1 = ax.bar(x - width/2, comparacion['P_empirica'],   width, label='Empírica',    color='steelblue')
bars2 = ax.bar(x + width/2, comparacion['P_NaiveBayes'], width, label='Naive Bayes', color='darkorange')
ax.set_xticks(x)
ax.set_xticklabels(comparacion['Transición'])
ax.set_ylabel('Probabilidad')
ax.set_title('Validación: Probabilidades Empíricas vs Naive Bayes')
ax.legend()
ax.set_ylim(0, 1)
for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', fontsize=9)
for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('outputs/validacion_probabilidades.png', dpi=150)
plt.show()



lambdas = {
    'λ_view\n(V→P)':     lambda_view,
    'λ_cart\n(P→C)':     lambda_cart,
    'λ_purchase\n(C→B)': lambda_purchase,
    'λ_abandon\n(P→A)':  lambda_abandon
}
colores = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(lambdas.keys(), lambdas.values(), color=colores, edgecolor='black', linewidth=0.7)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('λ (usuarios/hora)')
ax.set_title('Tasas de Transición λ — Input para el Sistema de EDOs')
plt.tight_layout()
plt.savefig('outputs/tasas_lambda_globales.png', dpi=150)
plt.show()



fig, ax = plt.subplots(figsize=(8, 5))
colores_esp = ['#e67e22', '#8e44ad', '#27ae60', '#2980b9', '#c0392b']
bars = ax.barh(df_segmentos['segmento'], df_segmentos['lambda_purchase'],
               color=colores_esp[:len(df_segmentos)], edgecolor='black', linewidth=0.7)
for bar in bars:
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.3f}', va='center', fontsize=9)
ax.set_xlabel('λ_purchase (adopciones/hora)')
ax.set_title('Tasa de Adopción Final por Especie')
plt.tight_layout()
plt.savefig('outputs/lambda_purchase_por_especie.png', dpi=150)
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(13, 4))
etapas = [
    ('Naive Bayes\n(Individuo)', ['p_view', 'p_cart', 'p_purchase'],
     [p_nb_view, p_nb_cart, p_nb_purchase], '#3498db'),
    ('Nexo\nλ = −ln(1−p̄)/τ', ['λ_view', 'λ_cart', 'λ_purchase'],
     [lambda_view, lambda_cart, lambda_purchase], '#e67e22'),
    ('EDOs\n(Sistema agregado)', ['dV/dt', 'dP/dt', 'dC/dt'],
     [None, None, None], '#2ecc71'),
]
for ax, (titulo, labels, vals, color) in zip(axes, etapas):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_facecolor('#f9f9f9')
    ax.text(0.5, 0.92, titulo, ha='center', va='top', fontsize=11,
            fontweight='bold', color=color,
            transform=ax.transAxes)
    for i, (label, val) in enumerate(zip(labels, vals)):
        y = 0.65 - i * 0.22
        rect = mpatches.FancyBboxPatch((0.1, y - 0.07), 0.8, 0.14,
                                        boxstyle="round,pad=0.02",
                                        linewidth=1.2, edgecolor=color,
                                        facecolor='white',
                                        transform=ax.transAxes)
        ax.add_patch(rect)
        txt = f"{label} = {val:.3f}" if val is not None else label
        ax.text(0.5, y, txt, ha='center', va='center', fontsize=9,
                transform=ax.transAxes)

# Flechas entre paneles
fig.text(0.355, 0.5, '→', fontsize=28, ha='center', va='center', color='gray')
fig.text(0.645, 0.5, '→', fontsize=28, ha='center', va='center', color='gray')
fig.suptitle('Nexo: Probabilidad Individual → Tasas → Dinámica Agregada',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/diagrama_nexo.png', dpi=150, bbox_inches='tight')
plt.show()