# LSP-AYNI

## Descripción del Proyecto

LSP-AYNI es un sistema de reconocimiento de lenguaje de señas Peruanas en tiempo real que utiliza un backend FastAPI con comunicación WebSocket, MediaPipe para extracción de puntos clave de manos, y TensorFlow para inferencia de aprendizaje automático. El sistema captura gestos de manos a través de webcam, extrae puntos clave y predice significados del lenguaje de señas.



## ARQUITECTURA DEL SISTEMA

### Diagrama de Arquitectura General

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FRONTEND      │    │    BACKEND       │    │   ML PIPELINE   │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Camera/Video│ │◄──►│ │ FastAPI      │ │◄──►│ │ MediaPipe   │ │
│ │ Capture     │ │    │ │ WebSocket    │ │    │ │ Keypoint    │ │
│ └─────────────┘ │    │ │ Server       │ │    │ │ Extraction  │ │
│                 │    │ └──────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │                  │    │                 │
│ │ WebSocket   │ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │ Client      │ │◄──►│ │ Video Upload │ │◄──►│ │ TensorFlow  │ │
│ └─────────────┘ │    │ │ Processor    │ │    │ │ Model       │ │
│                 │    │ └──────────────┘ │    │ │ Inference   │ │
│ ┌─────────────┐ │    │                  │    │ └─────────────┘ │
│ │ UI Controls │ │    │ ┌──────────────┐ │    │                 │
│ └─────────────┘ │    │ │ Model        │ │    │ ┌─────────────┐ │
└─────────────────┘    │ │ Processor    │ │◄──►│ │ Data        │ │
                       │ └──────────────┘ │    │ │ Normalization│ |
                       └──────────────────┘    │ └─────────────┘ │
                                               └─────────────────┘
```


## Comandos de Desarrollo

> [!CAUTION]
> Debes tener instalado la versión 3.10.10 de python para evitar errores de dependencias.


### Configuración y Gestión del Backend
```bash
# Instalar dependencias
cd backend
python utils.py --install-deps

# Ejecutar prueba completa del sistema
python utils.py --test

# Iniciar el servidor
python utils.py --start
# O directamente:
python main.py

# Verificar estado del sistema
python utils.py --info

# Verificar dependencias
python utils.py --check-deps
```

> [!NOTE]
> Más detalles del backend lo puedes encontrar [aquí](backend/BACKEND.md).

### Desarrollo Frontend
Los archivos estáticos son servidos por FastAPI. No se requiere proceso de compilación separado - los archivos se sirven directamente desde el directorio `static/`.

## Arquitectura del Sistema

### Componentes Principales
- **Backend FastAPI** (`backend/main.py`): Servidor WebSocket que maneja comunicación en tiempo real
- **Integración MediaPipe** (`backend/keypoint_extractor.py`): Extracción de puntos clave de manos usando MediaPipe
- **Procesamiento de Modelo ML** (`backend/model_processor.py`): Modelo TensorFlow para predicción de señas
- **Interfaz Frontend** (`static/`): Cliente JavaScript con integración de cámara WebRTC

### Flujo de Datos
1. El cliente web captura cuadros de video vía WebRTC
2. Los cuadros se envían al backend vía WebSocket como datos base64
3. MediaPipe extrae 42 puntos clave (21 por mano)
4. Los puntos clave se normalizan y procesan a través del modelo TensorFlow
5. Las predicciones se devuelven al cliente con puntuaciones de confianza

### Configuración Clave
- **Frames Objetivo**: 50 frames por secuencia de predicción
- **Duración de Grabación**: 2.5 segundos de captura
- **Cuenta Regresiva**: 3 segundos de tiempo de preparación
- **Ambas manos requeridas** para extracción válida de puntos clave
- **El servidor funciona en**: http://127.0.0.1:8000

## Modos de Desarrollo

### Modo Producción
Requiere archivos de modelo en el directorio `models/`:
- `modelo_finetuned_pucp_glosas.keras` (modelo TensorFlow)
- `label_encoder.pkl` (codificador de etiquetas scikit-learn)
- `model_info.pkl` (metadatos del modelo)

## Pruebas y Diagnósticos

El script `utils.py` proporciona pruebas integrales:
- Verificación de dependencias
- Prueba de acceso a cámara
- Prueba de funcionalidad MediaPipe
- Prueba de carga de modelo TensorFlow
- Validación de extractor de puntos clave
- Validación de procesador de modelo

## Dependencias

### Backend Python
- FastAPI + uvicorn para servidor web
- MediaPipe 0.10.8 para detección de manos
- TensorFlow 2.18.0 para inferencia ML
- OpenCV para procesamiento de imágenes
- WebSockets para comunicación en tiempo real

### Frontend
- JavaScript nativo (no se requieren herramientas de compilación)
- WebRTC para acceso a cámara
- WebSocket para comunicación con servidor