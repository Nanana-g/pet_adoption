# Endpoints disponibles — API Pet Adoption Funnel

Base URL: http://localhost:5000

| Método | Endpoint           | Descripción                                      |
|--------|--------------------|--------------------------------------------------|
| GET    | /api/parametros    | Parámetros λ base y tasas brutas del dataset     |
| GET    | /api/escenarios    | 4 escenarios predefinidos con series temporales  |
| GET    | /api/estadisticas  | Conversión por especie, device, source, campaign |
| POST   | /api/simular       | Simulación con parámetros personalizados         |
| POST   | /api/predecir      | Probabilidades individuales por usuario          |

## Formato de respuesta /api/simular y /api/escenarios
{
  "t": [...], <- eje de tiempo (horas)
  "V": [...], <- visitantes
  "P": [...], <- vistas de perfil
  "C": [...], <- solicitudes
  "B": [...], <- adopciones acumuladas
  "A": [...], <- abandonos acumulados
  "adopciones_fin":  <- número final de adopciones
  "abandonos_fin": <- número final de abandonos
  "conv_rate": <- tasa de conversión final (%)
}