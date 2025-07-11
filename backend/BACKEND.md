# LSP-VCOM Backend

Sistema de reconocimiento de lenguaje de señas Peruanas en tiempo real usando FastAPI, WebSocket y MediaPipe.



## 📋 Componentes

### 1. `keypoint_extractor.py`
- **Función**: Extrae keypoints de manos usando MediaPipe
- **Entrada**: Frames de video en base64
- **Salida**: 42 keypoints (21 por mano) normalizados
- **Funcionalidades**:
  - Detección de ambas manos requerida
  - Countdown de 3 segundos antes de grabar
  - Captura de 2.5 segundos de datos
  - Validación de calidad de keypoints

### 2. `model_processor.py`
- **Función**: Preprocesa datos y ejecuta predicciones
- **Procesos**:
  - Normalización de secuencia a 50 frames
  - Z-score normalization
  - Predicción con modelo TensorFlow
  - Post-procesamiento de resultados

### 3. `main.py`
- **Función**: Servidor FastAPI con WebSocket
- **Endpoints**:
  - `GET /`: Interfaz web principal
  - `GET /test`: Estado del servidor
  - `WebSocket /ws`: Comunicación en tiempo real
- **Modos**:
  - **Modo Producción**: Con modelo de IA completo
  - **Modo Demostración**: Predicciones simuladas

### 4. `config.py`
- **Función**: Configuración centralizada
- **Configuraciones**:
  - Rutas de archivos del modelo
  - Parámetros de MediaPipe
  - Configuración del servidor
  - Logging y debugging

### 5. `utils.py`
- **Función**: Utilidades para mantenimiento
- **Herramientas**:
  - Instalación de dependencias
  - Testing completo del sistema
  - Inicio del servidor
  - Diagnósticos

## 🚀 Instalación y Configuración

### 1. Instalar Dependencias

```bash
cd backend
python utils.py --install-deps
```

### 2. Verificar Sistema

```bash
python utils.py --test
```

### 3. Configurar Modelo (Opcional)

Si tienes el modelo entrenado, crea la carpeta `models` en el directorio raíz y coloca:

```
models/
├── modelo_finetuned_pucp_glosas.keras
├── label_encoder.pkl
└── model_info.pkl
```

### 4. Iniciar Servidor

```bash
python utils.py --start
```

O directamente:

```bash
python main.py
```

## 🔧 Uso del Sistema

### Flujo de Trabajo

1. **Inicio**: Usuario accede a la interfaz web
2. **Cámara**: Se solicita permiso de cámara
3. **Detección**: Sistema detecta ambas manos
4. **Countdown**: 3 segundos de preparación
5. **Grabación**: 2.5 segundos de captura de keypoints
6. **Procesamiento**: Normalización y preprocesamiento
7. **Predicción**: Modelo clasifica la seña
8. **Resultado**: Se muestra en la interfaz

### Protocolo WebSocket

#### Envío de Frame (Cliente → Servidor)
```json
{
  "type": "frame",
  "data": "base64_encoded_image",
  "settings": {
    "confidenceThreshold": 0.6,
    "predictionCount": 3,
    "frameRate": 66
  }
}
```

#### Respuesta de Predicción (Servidor → Cliente)
```json
{
  "hands_detected": true,
  "status": "✅ Predicción completada: Hola",
  "predictions": [
    {
      "rank": 1,
      "label": "Hola",
      "confidence": 0.92
    },
    {
      "rank": 2,
      "label": "Buenos días",
      "confidence": 0.78
    }
  ],
  "main_prediction": "Hola",
  "confidence": 0.92,
  "timestamp": 1672531200.0
}
```

## 📊 Estados del Sistema

### Estados de Captura
- `no_hands`: No se detectan ambas manos
- `hands_detected`: Ambas manos detectadas, listo para grabar
- `countdown`: Cuenta regresiva activa (3 segundos)
- `recording`: Grabando keypoints (4 segundos)
- `processing`: Procesando datos capturados

### Mensajes de Estado
- 👋 "Muestra ambas manos frente a la cámara"
- ✋ "Ambas manos detectadas - Listo para grabar"
- 🔥 "Preparándose para grabar... ¡Mantén tus manos visibles!"
- 🎬 "Grabando... X% completado"
- ✅ "Predicción completada: [SEÑA]"

## 🔍 Diagnóstico y Testing

### Pruebas Individuales

```bash
# Verificar dependencias
python utils.py --check-deps

# Probar cámara
python utils.py --check-camera

# Información del sistema
python utils.py --info
```

### Prueba Completa

```bash
python utils.py --test
```

Ejecuta:
- ✅ Verificación de dependencias
- ✅ Test de acceso a cámara
- ✅ Test de MediaPipe
- ✅ Test de TensorFlow
- ✅ Test de extractor de keypoints
- ✅ Test de procesador del modelo

## 📁 Estructura de Archivos

```
backend/
├── main.py                 # Servidor FastAPI principal
├── keypoint_extractor.py   # Extractor de keypoints MediaPipe
├── model_processor.py      # Preprocesamiento y modelo
├── config.py              # Configuración centralizada
├── utils.py               # Utilidades y testing
├── requirements.txt       # Dependencias Python
└── BACKEND.md             # Esta documentación
```

## 🌐 URLs del Servidor

- **Interfaz Web**: http://127.0.0.1:8000/
- **API Test**: http://127.0.0.1:8000/test
- **WebSocket**: ws://127.0.0.1:8000/ws
- **Archivos Estáticos**: http://127.0.0.1:8000/static/

## 📈 Métricas y Performance

### Latencia Esperada
- **Extracción Keypoints**: ~10-30ms
- **Preprocesamiento**: ~5-15ms
- **Predicción Modelo**: ~50-200ms
- **Total por Predicción**: ~100-300ms

### Recursos del Sistema
- **RAM**: ~2-4GB (con modelo cargado)
- **CPU**: Moderate (depende de resolución de cámara)
- **GPU**: Opcional (mejora performance de TensorFlow)

## 🔒 Seguridad y Privacidad

- Los frames de video se procesan localmente
- No se almacenan imágenes ni videos
- Solo se extraen keypoints normalizados
- Comunicación WebSocket sin persistencia

## 🎯 Limitaciones Conocidas

1. **Requiere ambas manos visibles** durante toda la secuencia
2. **Iluminación adecuada** para detección de MediaPipe
3. **Fondo contrastante** recomendado
4. **Distancia óptima** de la cámara (aprox. 60-80cm)
5. **Resolución mínima** de cámara 640x480

## 🚀 Próximas Mejoras

- Soporte para una sola mano
- Mejora en condiciones de iluminación
- Optimización de performance
- Grabación de datos para reentrenamiento
- Dashboard de administración

---

**Versión**: 1.0.0  
**Fecha**: Junio 2025  
**Autor**: Sistema LSP-VCOM

