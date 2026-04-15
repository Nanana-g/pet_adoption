'''
App para el backend
autor: Ariana Víquez

pasos para iniciar:
- Haber ejecurtado todos los notebooks en el orden especificado (empezando por el naive_bayes.py)
-Ejecutar en la terminal en el siguiente orden:
    - cd backend
    - python app.py

- opcionales, verificar funcionamiento con los curls
curl http://localhost:5000/api/parametros

curl http://localhost:5000/api/escenarios

curl http://localhost:5000/api/estadisticas

curl -X POST http://localhost:5000/api/simular \
  -H "Content-Type: application/json" \
  -d '{"lambda_view": 1.2, "alpha_mult": 2.0, "alpha_spike": true}'

curl -X POST http://localhost:5000/api/predecir \
  -H "Content-Type: application/json" \
  -d '{"species": "dog", "device": "desktop", "returning_user": 1, "discount_seen": 1}'
'''


from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.integrate import solve_ivp
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


print("Cargando datos y entrenando modelos...")

df_raw = pd.read_csv('data/pet_adoption_funnel.csv')
tasas  = pd.read_csv('outputs/tasas_lambda.csv').set_index('parametro')['valor']

LAMBDA_VIEW= float(tasas['lambda_view'])
LAMBDA_CART= float(tasas['lambda_cart'])
LAMBDA_PURCHASE= float(tasas['lambda_purchase'])
LAMBDA_ABANDON= float(tasas['lambda_abandon'])
N = len(df_raw)
T_HORAS = 72
ALPHA_BASE = N / T_HORAS

#encoding
df_enc = df_raw.copy()
cat_cols = ['species', 'breed', 'age_months', 'source', 'device', 'weekday', 'campaign_type']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_raw[col])
    encoders[col] = le

FEATURES = [
    'species', 'breed', 'age_months', 'source', 'device', 'weekday',
    'returning_user', 'discount_seen', 'campaign_type',
    'session_time', 'pages_viewed', 'shelter_rating',
    'distance_km', 'vet_check_available', 'neutered'
]

#entrenar los 3 modelos 
X1 = df_enc[FEATURES]
y1 = df_enc['viewed_product']
modelo_view = GaussianNB().fit(X1, y1)

mask2 = df_enc['viewed_product'] == 1
X2 = df_enc.loc[mask2, FEATURES]
y2 = df_enc.loc[mask2, 'added_to_cart']
modelo_cart = GaussianNB().fit(X2, y2)

mask3 = df_enc['added_to_cart'] == 1
X3 = df_enc.loc[mask3, FEATURES]
y3 = df_enc.loc[mask3, 'purchased']
modelo_purchase = GaussianNB().fit(X3, y3)

print("Modelos correctamente entrenados, servidor listo")

#funcion central del sistema edos
def funnel_odes(t, y, lv, lc, lp, la, alpha_func):
    V, P, C, B, A = y
    alpha = alpha_func(t)
    dV = alpha - lv * V
    dP = lv * V - lc * P - la * P
    dC = lc * P - lp * C
    dB = lp * C
    dA = la * P
    return [dV, dP, dC, dB, dA]

def resolver_odes(lv, lc, lp, la, alpha_func, t_horas=72, puntos=300):
    """
    Resuelve el sistema de EDOs y retorna un dict serializable a JSON.
    """
    V0 = N * (1 - df_raw['viewed_product'].mean())
    P0 = df_raw['viewed_product'].sum() - df_raw['added_to_cart'].sum()
    C0 = df_raw['added_to_cart'].sum()  - df_raw['purchased'].sum()
    B0 = float(df_raw['purchased'].sum())
    A0 = float(df_raw['abandoned'].sum())
    y0 = [V0, P0, C0, B0, A0]


    t_eval = np.linspace(0, t_horas, puntos)



    sol = solve_ivp(
        fun    = lambda t, y: funnel_odes(t, y, lv, lc, lp, la, alpha_func),
        t_span = (0, t_horas),
        y0     = y0,
        t_eval = t_eval,
        method = 'RK45',
        rtol   = 1e-6,
        atol   = 1e-8
    )


    V, P, C, B, A = sol.y
    conv_rate = float(B[-1] / (B[-1] + A[-1]) * 100) if (B[-1] + A[-1]) > 0 else 0

    return {
        't': t_eval.tolist(),
        'V': V.tolist(),
        'P': P.tolist(),
        'C': C.tolist(),
        'B': B.tolist(),
        'A': A.tolist(),
        'adopciones_fin': round(float(B[-1]), 1),
        'abandonos_fin': round(float(A[-1]), 1),
        'conv_rate': round(conv_rate, 2),
    }

# endpoints de funcion

@app.route('/api/parametros', methods=['GET'])
def get_parametros(): #retorna los parámetros base del modelo para que la UI
    return jsonify({
        'lambda_view': round(LAMBDA_VIEW,     4),
        'lambda_cart': round(LAMBDA_CART,     4),
        'lambda_purchase': round(LAMBDA_PURCHASE, 4),
        'lambda_abandon': round(LAMBDA_ABANDON,  4),
        'alpha_base': round(ALPHA_BASE,      2),
        't_horas': T_HORAS,
        'n_usuarios': N,
        'tasas_brutas': {
            'p_view': round(float(df_raw['viewed_product'].mean()), 4),
            'p_cart': round(float(df_raw['added_to_cart'].mean()),  4),
            'p_purchase': round(float(df_raw['purchased'].mean()),      4),
            'p_abandon': round(float(df_raw['abandoned'].mean()),      4),
        }
    })


@app.route('/api/simular', methods=['POST'])
def simular(): # recibe los parametros para retornar el funnel
    data = request.get_json()

    lv = float(data.get('lambda_view', LAMBDA_VIEW))
    lc = float(data.get('lambda_cart', LAMBDA_CART))
    lp = float(data.get('lambda_purchase', LAMBDA_PURCHASE))
    la = float(data.get('lambda_abandon', LAMBDA_ABANDON))
    alpha_mult  = float(data.get('alpha_mult', 1.0))
    alpha_spike = bool(data.get('alpha_spike', False))
    spike_mult  = float(data.get('spike_mult', 3.0))
    t_horas = int(data.get('t_horas',       T_HORAS))

    alpha_base_custom = ALPHA_BASE * alpha_mult

    if alpha_spike:
        def alpha_func(t):
            return alpha_base_custom * spike_mult if t <= 24 else alpha_base_custom
    else:
        def alpha_func(t):
            return alpha_base_custom

    resultado = resolver_odes(lv, lc, lp, la, alpha_func, t_horas)
    resultado['parametros_usados'] = {
        'lambda_view': lv, 'lambda_cart': lc,
        'lambda_purchase': lp, 'lambda_abandon': la,
        'alpha_mult': alpha_mult, 'alpha_spike': alpha_spike,
        'spike_mult': spike_mult, 't_horas': t_horas
    }
    return jsonify(resultado)



@app.route('/api/escenarios', methods=['GET'])
def get_escenarios():
    escenarios = {}

    #esce 1
    escenarios['base'] = resolver_odes(
        LAMBDA_VIEW, LAMBDA_CART, LAMBDA_PURCHASE, LAMBDA_ABANDON,
        lambda t: ALPHA_BASE
    )
    escenarios['base']['nombre'] = 'Base — Sin intervención'

    #esce 2
    def alpha_campana(t):
        return ALPHA_BASE * 3.0 if t <= 24 else ALPHA_BASE
    escenarios['campana'] = resolver_odes(
        LAMBDA_VIEW,
        LAMBDA_CART * 1.20,
        LAMBDA_PURCHASE * 1.20,
        LAMBDA_ABANDON * 0.85,
        alpha_campana
    )
    escenarios['campana']['nombre'] = 'Campaña de Descuento'

    #esce 3
    MOBILE_SHARE = 0.55
    MEJORA_UX = 0.35
    escenarios['ux_movil'] = resolver_odes(
        LAMBDA_VIEW,
        LAMBDA_CART * (1 + MOBILE_SHARE * MEJORA_UX),
        LAMBDA_PURCHASE * (1 + MOBILE_SHARE * MEJORA_UX),
        LAMBDA_ABANDON * (1 - MOBILE_SHARE * MEJORA_UX * 0.5),
        lambda t: ALPHA_BASE
    )
    escenarios['ux_movil']['nombre'] = 'Mejora UX Móvil'

    #esce 4
    def alpha_crisis(t):
        return ALPHA_BASE if t <= 12 else ALPHA_BASE * 0.60
    escenarios['crisis'] = resolver_odes(
        LAMBDA_VIEW * 0.70,
        LAMBDA_CART * 0.80,
        LAMBDA_PURCHASE * 0.75,
        LAMBDA_ABANDON * 1.50,
        alpha_crisis
    )
    escenarios['crisis']['nombre'] = 'Crisis de Reputación'

    return jsonify(escenarios)


@app.route('/api/predecir', methods=['POST'])
def predecir(): #posibilidades de transitar segun el funnel
    data = request.get_json()

    #encoding
    def encode_field(field, value):
        le = encoders[field]
        if value in le.classes_:
            return int(le.transform([value])[0])
        else:
            return 0   # valor desconocido → clase 0 por defecto

    try:
        fila = pd.DataFrame([{
            'species': encode_field('species', data.get('species', 'dog')),
            'breed': encode_field('breed', data.get('breed', 'Mixed')),
            'age_months': encode_field('age_months', data.get('age_months', '3-6')),
            'source': encode_field('source', data.get('source', 'organic')),
            'device': encode_field('device', data.get('device', 'mobile')),
            'weekday': encode_field('weekday', data.get('weekday', 'Monday')),
            'returning_user': int(data.get('returning_user', 0)),
            'discount_seen': int(data.get('discount_seen', 0)),
            'campaign_type': encode_field('campaign_type', data.get('campaign_type', 'none')),
            'session_time': float(data.get('session_time', 3.0)),
            'pages_viewed': int(data.get('pages_viewed', 2)),
            'shelter_rating': float(data.get('shelter_rating', 4.0)),
            'distance_km': float(data.get('distance_km', 10.0)),
            'vet_check_available': int(data.get('vet_check_available', 1)),
            'neutered': int(data.get('neutered', 1)),
        }])


        p_view = float(modelo_view.predict_proba(fila)[0][1])
        p_cart = float(modelo_cart.predict_proba(fila)[0][1])
        p_purchase = float(modelo_purchase.predict_proba(fila)[0][1])
        p_abandon = 1 - p_purchase



        p_full_funnel = p_view * p_cart * p_purchase

        return jsonify({
            'p_view': round(p_view, 4),
            'p_cart': round(p_cart, 4),
            'p_purchase': round(p_purchase, 4),
            'p_abandon': round(p_abandon, 4),
            'p_full_funnel': round(p_full_funnel, 4),
            'perfil_recibido': data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400



@app.route('/api/estadisticas', methods=['GET'])
def get_estadisticas(): #estadísticas agregadas del dataset para los paneles

    stats = {
        'total_usuarios': N,
        'por_especie': df_raw.groupby('species').agg(
            n=('user_id', 'count'),
            conv_rate=('purchased', 'mean')
        ).round(4).reset_index().to_dict(orient='records'),
        'por_device': df_raw.groupby('device').agg(
            n=('user_id', 'count'),
            conv_rate=('purchased', 'mean')
        ).round(4).reset_index().to_dict(orient='records'),
        'por_source': df_raw.groupby('source').agg(
            n=('user_id', 'count'),
            conv_rate=('purchased', 'mean')
        ).round(4).reset_index().to_dict(orient='records'),
        'por_campaign': df_raw.groupby('campaign_type').agg(
            n=('user_id', 'count'),
            conv_rate=('purchased', 'mean')
        ).round(4).reset_index().to_dict(orient='records'),
        'funnel_global': {
            'viewed_product': round(float(df_raw['viewed_product'].mean()), 4),
            'added_to_cart':  round(float(df_raw['added_to_cart'].mean()), 4),
            'purchased': round(float(df_raw['purchased'].mean()), 4),
            'abandoned': round(float(df_raw['abandoned'].mean()), 4),
        }
    }
    return jsonify(stats)

#Arrancar el servidor
if __name__ == '__main__':
    app.run(debug=True, port=5000)

