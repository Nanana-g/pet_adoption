'''
code de naive bayes
autor: Ariana Víquez

Ejecutar primero
'''

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#─Cargar datos
df = pd.read_csv('data/pet_adoption_funnel.csv')
print(df.shape)
print(df.head())
print(df.dtypes)

#──Encoding 
le= LabelEncoder()
cat_cols= ['species', 'breed', 'age_months', 'source', 'device', 'weekday', 'campaign_type']

df_encoded= df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df[col])

print(df_encoded[cat_cols].head())

#──Features comuness
FEATURES= [
    'species', 'breed', 'age_months', 'source', 'device', 'weekday',
    'returning_user', 'discount_seen', 'campaign_type',
    'session_time', 'pages_viewed', 'shelter_rating',
    'distance_km', 'vet_check_available', 'neutered'
]

#visita → view de perfil
X1 = df_encoded[FEATURES]
y1 = df_encoded['viewed_product']

#vista → soli
mask2 = df_encoded['viewed_product'] == 1
X2 = df_encoded.loc[mask2, FEATURES]
y2 = df_encoded.loc[mask2, 'added_to_cart']

#solicitud → adopt
mask3 = df_encoded['added_to_cart'] == 1
X3 = df_encoded.loc[mask3, FEATURES]
y3 = df_encoded.loc[mask3, 'purchased']

#entrenamiento y eva
def entrenar_nb(X, y, nombre):
    X_train, X_test, y_train, y_test= train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo= GaussianNB()
    modelo.fit(X_train, y_train)
    y_pred= modelo.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"  Modelo: {nombre}")
    print(f"{'~'*50}")
    print(classification_report(y_test, y_pred))
    
    #matriz de confusión
    cm= confusion_matrix(y_test, y_pred)
    disp= ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Sí'])
    disp.plot(cmap='Oranges')
    plt.title(f'Matriz de confusión — {nombre}')
    plt.tight_layout()
    plt.savefig(f'outputs/cm_{nombre.replace(" ", "_")}.png', dpi=150)
    plt.show()
    
    return modelo

modelo_1= entrenar_nb(X1, y1, "Visita a vista de Perfil")
modelo_2= entrenar_nb(X2, y2, "Vista a solicitud")
modelo_3= entrenar_nb(X3, y3, "Solicitud a adopcion")


#prob individuales P
df_encoded['p_view']     = modelo_1.predict_proba(df_encoded[FEATURES])[:, 1]

df_encoded['p_cart']     = np.nan
df_encoded.loc[mask2, 'p_cart'] = modelo_2.predict_proba(X2)[:, 1]

df_encoded['p_purchase'] = np.nan
df_encoded.loc[mask3, 'p_purchase'] = modelo_3.predict_proba(X3)[:, 1]

#downlo
df_encoded[['user_id', 'species', 'source', 'device',
            'p_view', 'p_cart', 'p_purchase']].to_csv(
    'outputs/probabilidades_individuales.csv', index=False
)

print("\nProbabilidades promedio por especie:")
print(df_encoded.groupby('species')[['p_view', 'p_cart', 'p_purchase']].mean().round(3))


#visualización del modelo 1
feature_importance= pd.DataFrame({
    'feature': FEATURES,
    'mean_class_0': modelo_1.theta_[0],
    'mean_class_1': modelo_1.theta_[1],
})
feature_importance['diferencia'] = abs(
    feature_importance['mean_class_1'] - feature_importance['mean_class_0']
)
feature_importance= feature_importance.sort_values('diferencia', ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(feature_importance['feature'], feature_importance['diferencia'], color='darkorange')
plt.xlabel('Diferencia de medias entre clases (proxy de importancia)')
plt.title('Variables más discriminantes — Transición 1: Visita → Vista de Perfil')
plt.tight_layout()
plt.savefig('outputs/feature_importance_t1.png', dpi=150)
plt.show()

