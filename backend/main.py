from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
import uvicorn
from pathlib import Path
import os
import sys
import numpy as np

# Agregar el directorio backend al path para imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keypoint_extractor import HandKeypointExtractor
from model_processor import SignLanguageModel, ModelPreprocessor
from config import (
    MODEL_CONFIG, SERVER_CONFIG, LOGGING_CONFIG, 
    ensure_directories, check_model_files
)

# Configurar logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# Configuración de la aplicación
app = FastAPI(
    title="LSP-AYNI API",
    description="API para reconocimiento de lenguaje de señas PUCP-GLOSAS en tiempo real",
    version="1.0.0"
)

# Variables globales
keypoint_extractor = None
sign_model = None

class VideoUploadProcessor:
    """Procesa frames de video upload y acumula keypoints"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.keypoints_buffer = []
        self.total_frames = 0
        self.frames_with_hands = 0
        self.target_frames = 50
    
    def add_frame(self, keypoints):
        """Añade keypoints de un frame"""
        self.total_frames += 1
        if keypoints is not None:
            self.keypoints_buffer.append(keypoints.copy())
            self.frames_with_hands += 1
    
    def is_ready_for_prediction(self):
        """Verifica si tenemos suficientes frames para predicción"""
        return len(self.keypoints_buffer) >= self.target_frames  # 50 frames para auto-predicción
    
    def get_keypoints_sequence(self):
        """Retorna la secuencia de keypoints para el modelo"""
        if len(self.keypoints_buffer) >= 21:  # Permitir mínimo 21 frames
            # Tomar hasta 50 frames o los que tengamos
            frames_to_use = min(len(self.keypoints_buffer), self.target_frames)
            keypoints_array = np.array(self.keypoints_buffer[:frames_to_use])
            
            # Si tenemos menos de 30 frames, advertir pero permitir predicción
            if len(self.keypoints_buffer) < 30:
                print(f"⚠️ VIDEO UPLOAD: Predicción con solo {len(self.keypoints_buffer)} frames (recomendado: 30+)")
            
            return keypoints_array
        return None
    
    def get_progress(self):
        """Retorna el progreso de extracción"""
        return {
            'total_frames': self.total_frames,
            'frames_with_hands': self.frames_with_hands,
            'target_frames': self.target_frames,
            'progress_percent': (self.frames_with_hands / self.target_frames) * 100
        }
    
    def should_process_final(self, total_video_frames):
        """Verifica si debe procesar al final del video"""
        return (self.total_frames >= total_video_frames and 
                len(self.keypoints_buffer) >= 21 and  # Reducido a 21 mínimo
                not self.is_ready_for_prediction())

# Instancia global para video upload
video_upload_processor = VideoUploadProcessor()

# Estado anterior para evitar log spam
previous_main_log_state = None

# Control de flujos mutuamente excluyentes
is_processing_video_upload = False
video_upload_timeout_task = None

class ConnectionManager:
    """Manejador de conexiones WebSocket"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nueva conexión WebSocket. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Conexión WebSocket cerrada. Total: {len(self.active_connections)}")
        
    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error enviando mensaje WebSocket: {e}")
            self.disconnect(websocket)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Inicialización de la aplicación"""
    global keypoint_extractor, sign_model
    
    logger.info("🚀 Iniciando LSP-AYNI API Server...")
    
    # Crear directorios necesarios
    ensure_directories()
    
    # Inicializar extractor de keypoints
    try:
        keypoint_extractor = HandKeypointExtractor()
        logger.info("✅ Extractor de keypoints inicializado")
    except Exception as e:
        logger.error(f"❌ Error inicializando extractor: {e}")
        keypoint_extractor = None
    
    # Intentar cargar modelo
    try:
        # Verificar si existen los archivos del modelo
        model_files_exist = check_model_files()
        
        if model_files_exist:
            sign_model = SignLanguageModel(
                model_path=str(MODEL_CONFIG["model_path"]),
                encoder_path=str(MODEL_CONFIG["encoder_path"]),
                info_path=str(MODEL_CONFIG["info_path"])
            )
            
            if sign_model.load_model_components():
                logger.info("✅ Modelo de IA cargado exitosamente")
            else:
                logger.error("❌ Error cargando modelo - Servidor no puede funcionar sin modelo")
                raise Exception("Modelo no pudo ser cargado")
        else:
            logger.error("❌ Archivos del modelo no encontrados - Servidor no puede funcionar sin modelo")
            raise Exception("Archivos del modelo no encontrados")
            
    except Exception as e:
        logger.error(f"❌ Error configurando modelo: {e}")
        raise e
    
    logger.info("🤖 Servidor iniciado con modelo de IA completo")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la aplicación"""
    global keypoint_extractor
    
    logger.info("🛑 Cerrando LSP-AYNI API Server...")
    
    if keypoint_extractor:
        keypoint_extractor.cleanup()
        logger.info("✅ Extractor de keypoints limpiado")

# Servir archivos estáticos
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"📁 Sirviendo archivos estáticos desde: {static_dir}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Página principal"""
    try:
        index_path = static_dir / "index.html"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(
                content="<h1>LSP-AYNI API</h1><p>Interfaz web no encontrada</p>"
            )
    except Exception as e:
        logger.error(f"Error sirviendo página principal: {e}")
        return HTMLResponse(content="<h1>Error del servidor</h1>")

@app.get("/test")
async def test_endpoint():
    """Endpoint de prueba para verificar estado del servidor"""
    global sign_model, keypoint_extractor
    
    status = {
        "server": "LSP-AYNI API",
        "status": "running",
        "timestamp": time.time(),
        "components": {
            "keypoint_extractor": keypoint_extractor is not None,
            "sign_model": sign_model is not None and sign_model.is_ready() if sign_model else False
        }
    }
    
    if sign_model:
        status["model_info"] = sign_model.get_model_info()
    
    return JSONResponse(content=status)

@app.get("/api/model/info")
async def get_model_info():
    """Información del modelo cargado"""
    global sign_model
    
    if sign_model and sign_model.is_ready():
        return JSONResponse(content=sign_model.get_model_info())
    else:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

async def process_frame_with_model(base64_data: str, settings: dict) -> dict:
    """
    Procesa un frame usando el modelo de IA real
    
    Args:
        base64_data: Frame en base64
        settings: Configuración del cliente
        
    Returns:
        Diccionario con resultados del procesamiento
    """
    global keypoint_extractor, sign_model
    
    if not keypoint_extractor:
        return {"error": "Extractor de keypoints no disponible"}
    
    # Procesar frame
    hands_detected, keypoints, status = keypoint_extractor.process_base64_frame(base64_data)
    
    # Debug: Log solo en transiciones importantes (no durante countdown/recording)
    global previous_main_log_state
    
    # Solo log cuando se inicia el countdown o la grabación
    if status == "hands_detected" and not keypoint_extractor.is_recording and not keypoint_extractor.countdown_active:
        if previous_main_log_state != "hands_detected_initial":
            print(f"🔍 MAIN DEBUG: hands_detected={hands_detected}, status='{status}'")
            print(f"🔍 MAIN DEBUG: is_recording={keypoint_extractor.is_recording}, countdown_active={keypoint_extractor.countdown_active}, is_paused={keypoint_extractor.is_paused}")
            previous_main_log_state = "hands_detected_initial"
    
    response = {
        "hands_detected": hands_detected,
        "status": keypoint_extractor.get_status_message(hands_detected),
        "source": "camera",  # Identificar explícitamente como mensaje de cámara
        "timestamp": time.time()
    }
    
    # Manejar estados del extractor
    if status.startswith("paused:"):
        # En pausa después de predicción
        remaining_time = float(status.split(":")[1])
        response["status"] = f"⏸️ Pausa después de predicción... {remaining_time:.1f}s"
        response["paused"] = True
        
    elif status == "pause_ended":
        # Pausa terminó, volver al estado normal
        response["status"] = keypoint_extractor.get_status_message(hands_detected)
        response["paused"] = False
        
    elif status == "hands_detected" and not keypoint_extractor.is_recording and not keypoint_extractor.countdown_active and not keypoint_extractor.is_paused:
        # Iniciar countdown cuando se detecten ambas manos (solo si no estamos en pausa)
        start_state = f"starting_countdown:{status}"
        if previous_main_log_state != start_state:
            print(f"🚀 MAIN DEBUG: ¡INICIANDO COUNTDOWN! Status: {status}")
            previous_main_log_state = start_state
        asyncio.create_task(start_recording_sequence())
        response["status"] = "🔥 Iniciando secuencia de grabación..."
        
    elif status.startswith("countdown:") and keypoint_extractor.countdown_active:
        # Durante el countdown - SOLO si countdown está realmente activo
        remaining = status.split(":")[1]
        if int(remaining) > 0:  # Solo enviar si hay tiempo restante
            response["countdown_active"] = True
            response["countdown_remaining"] = int(remaining)
            response["status"] = f"🔥 Iniciando grabación en {remaining} segundos..."
            # Solo log una vez por cada segundo del countdown
            countdown_log_state = f"countdown_log:{remaining}"
            if previous_main_log_state != countdown_log_state:
                print(f"⏰ COUNTDOWN: {remaining} segundos restantes")
                previous_main_log_state = countdown_log_state
            #print(f"📤 ENVIANDO AL FRONTEND: countdown_active=True, countdown_remaining={remaining}")
        
    elif status.startswith("recording:"):
        progress = float(status.split(":")[1])
        response["recording_progress"] = progress
        
        # Log solo al inicio de la grabación
        if previous_main_log_state != "recording_active" and progress < 0.05:
            print(f"🎥 RECORDING: Grabando...")
            previous_main_log_state = "recording_active"
        
        # Verificar si debe terminar la grabación
        if keypoint_extractor.should_stop_recording():
            captured_keypoints = keypoint_extractor.stop_recording()
            if captured_keypoints is not None:
                frame_count = captured_keypoints.shape[0] if len(captured_keypoints.shape) > 0 else 0
                
                # Log al completar la grabación
                print(f"✅ GRABACIÓN COMPLETADA: {frame_count} frames")
                previous_main_log_state = "recording_completed"
                
                # Validar frame count mínimo (21 frames para entrada del modelo)
                if frame_count < 21:
                    response.update({
                        "error": f"Frames insuficientes: {frame_count}/21. El modelo requiere al menos 21 frames.",
                        "frame_count": frame_count,
                        "status": f"❌ Solo {frame_count} frames capturados (se requieren: 21)",
                        "hands_detected": False
                    })
                    # logger.warning(f"❌ Frames insuficientes: {frame_count}/40 - Rechazando predicción")  # Comentado
                else:
                    # Procesar con el modelo
                    prediction_result = await process_keypoints_with_model(captured_keypoints, settings)
                    # Agregar frame count a la respuesta exitosa
                    prediction_result["frame_count"] = frame_count
                    response.update(prediction_result)
                
                # Iniciar pausa después de la predicción o error
                keypoint_extractor.start_pause()
    
    return response

async def start_recording_sequence():
    """Inicia la secuencia de countdown + grabación"""
    global keypoint_extractor
    
    if keypoint_extractor:
        # Countdown de 3 segundos
        countdown_success = await keypoint_extractor.start_countdown()
        
        if countdown_success:
            # Iniciar grabación
            keypoint_extractor.start_recording()

async def process_video_upload_frame(base64_data: str, settings: dict) -> dict:
    """
    Procesa un frame de video upload de manera directa
    
    Args:
        base64_data: Frame en base64
        settings: Configuración del cliente
        
    Returns:
        Diccionario con resultados del procesamiento
    """
    global keypoint_extractor, sign_model, video_upload_processor
    
    if not keypoint_extractor:
        return {"error": "Extractor de keypoints no disponible"}
    
    # Cambiar a modo upload para suprimir logs de cámara
    keypoint_extractor.processing_mode = "upload"
    
    # Procesar frame directamente (sin countdown ni grabación)
    hands_detected, keypoints, status = keypoint_extractor.process_base64_frame(base64_data)
    
    # Restaurar modo camera para próximos frames de cámara
    keypoint_extractor.processing_mode = "camera"
    
    # Añadir frame al procesador
    video_upload_processor.add_frame(keypoints if hands_detected else None)
    
    # Obtener progreso actual
    progress = video_upload_processor.get_progress()
    
    response = {
        "hands_detected": hands_detected,
        "source": "upload",
        "timestamp": time.time(),
        "upload_progress": progress
    }
    
    if hands_detected and keypoints is not None:
        print(f"📹 VIDEO UPLOAD: Frame {progress['total_frames']} - Keypoints extraídos ({progress['frames_with_hands']}/{progress['target_frames']})")
        response["status"] = f"✅ Frame {progress['total_frames']} procesado - {progress['frames_with_hands']}/{progress['target_frames']} frames válidos"
        response["keypoints_extracted"] = True
    else:
        print(f"📹 VIDEO UPLOAD: Frame {progress['total_frames']} - Sin manos detectadas")
        response["status"] = f"⚠️ Frame {progress['total_frames']} - No se detectaron ambas manos"
        response["keypoints_extracted"] = False
    
    # Verificar si tenemos suficientes frames para predicción
    if video_upload_processor.is_ready_for_prediction():
        print(f"🎯 VIDEO UPLOAD: ¡{progress['target_frames']} frames recolectados! Realizando predicción...")
        
        # Obtener secuencia de keypoints
        keypoints_sequence = video_upload_processor.get_keypoints_sequence()
        
        if keypoints_sequence is not None:
            # Procesar con el modelo
            prediction_result = await process_keypoints_with_model(keypoints_sequence, settings)
            response.update(prediction_result)
            response["hands_detected"] = True  # Predicción exitosa implica detección
            response["status"] = f"✅ Predicción completada: {prediction_result.get('main_prediction', 'Error')}"
            
            print(f"🔍 DEBUG AUTO-PREDICCIÓN: Enviando response con source='{response.get('source')}' y status='{response.get('status')}'")
            print(f"🔍 DEBUG AUTO-PREDICCIÓN: hands_detected={response.get('hands_detected')}")
            
            # Reset processor para próximo video
            video_upload_processor.reset()
            
            # IMPORTANTE: Restaurar funcionalidad de cámara al completar automáticamente
            global is_processing_video_upload, video_upload_timeout_task
            is_processing_video_upload = False
            
            # Cancelar timeout de seguridad
            if video_upload_timeout_task and not video_upload_timeout_task.done():
                video_upload_timeout_task.cancel()
            
            print("📷 CÁMARA: Restaurada funcionalidad - Video upload auto-completado")
        else:
            response["error"] = "Error obteniendo secuencia de keypoints"
    
    return response

async def process_keypoints_with_model(keypoints, settings: dict) -> dict:
    """
    Procesa keypoints capturados con el modelo
    
    Args:
        keypoints: Array de keypoints capturados
        settings: Configuración del cliente
        
    Returns:
        Diccionario con predicciones
    """
    global sign_model
    
    if not sign_model or not sign_model.is_ready():
        return {"error": "Modelo no disponible"}
    
    try:
        # Realizar predicción
        result = sign_model.predict(keypoints, top_k=settings.get('predictionCount', 3))
        
        if result:
            # Filtrar por umbral de confianza
            threshold = settings.get('confidenceThreshold', 0.6)
            filtered_predictions = [
                p for p in result['predictions'] 
                if p['confidence'] >= threshold
            ]
            
            return {
                "predictions": filtered_predictions,
                "main_prediction": result['main_prediction'],
                "confidence": result['confidence'],
                "status": f"✅ Predicción completada: {result['main_prediction']}"
            }
        else:
            return {"error": "Error en predicción del modelo"}
            
    except Exception as e:
        logger.error(f"Error procesando con modelo: {e}")
        return {"error": f"Error en procesamiento: {str(e)}"}

async def reset_video_upload_after_timeout():
    """Reset video upload después de timeout de seguridad"""
    await asyncio.sleep(30)  # 30 segundos timeout
    global is_processing_video_upload, video_upload_processor
    if is_processing_video_upload:
        print("⏰ VIDEO UPLOAD: Timeout alcanzado - Restaurando funcionalidad de cámara")
        is_processing_video_upload = False
        video_upload_processor.reset()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint principal de WebSocket para comunicación en tiempo real"""
    global video_upload_processor, is_processing_video_upload, video_upload_timeout_task
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                # Procesar frame recibido
                base64_data = message.get("data", "")
                settings = message.get("settings", {})
                source = message.get("source", "camera")  # Default: camera
                
                if source == "upload":
                    # Solo procesar video upload si no hay cámara activa
                    if not is_processing_video_upload:
                        # Iniciar modo video upload
                        is_processing_video_upload = True
                        print("🎬 VIDEO UPLOAD: Iniciando procesamiento de video - Pausando cámara")
                    
                    # Procesamiento directo para video upload (sin countdown/grabación)
                    result = await process_video_upload_frame(base64_data, settings)
                else:
                    # Solo procesar cámara si no hay video upload activo
                    if is_processing_video_upload:
                        # Rechazar frames de cámara durante video upload
                        result = {
                            "hands_detected": False,
                            "status": "📹 Procesando video upload - Cámara pausada temporalmente",
                            "camera_paused": True,
                            "source": "camera_blocked",  # Identificar como cámara bloqueada
                            "timestamp": time.time()
                        }
                    else:
                        # Procesamiento normal para cámara (con countdown/grabación)
                        result = await process_frame_with_model(base64_data, settings)
                
                # Enviar respuesta
                await manager.send_message(websocket, result)
                
            elif message.get("type") == "ping":
                # Responder ping para mantener conexión
                await manager.send_message(websocket, {"type": "pong", "timestamp": time.time()})
                
            elif message.get("type") == "reset_video_upload":
                # Reset del procesador de video upload
                video_upload_processor.reset()
                is_processing_video_upload = True  # Activar modo video upload
                
                # Cancelar timeout anterior si existe
                if video_upload_timeout_task and not video_upload_timeout_task.done():
                    video_upload_timeout_task.cancel()
                
                # Iniciar nuevo timeout de seguridad
                video_upload_timeout_task = asyncio.create_task(reset_video_upload_after_timeout())
                
                print("🔄 VIDEO UPLOAD: Procesador reseteado para nuevo video - Pausando cámara")
                await manager.send_message(websocket, {
                    "type": "video_upload_reset", 
                    "status": "✅ Procesador reseteado - Listo para nuevo video"
                })
                
            elif message.get("type") == "video_upload_finished":
                # Procesar video upload final si tiene frames suficientes
                total_frames = message.get("total_frames", 50)
                print(f"🎬 VIDEO UPLOAD: Recibido mensaje de finalización - {len(video_upload_processor.keypoints_buffer)} frames válidos de {total_frames} total")
                
                if video_upload_processor.should_process_final(total_frames):
                    print(f"🎯 VIDEO UPLOAD: Finalizando con {len(video_upload_processor.keypoints_buffer)} frames válidos")
                    
                    keypoints_sequence = video_upload_processor.get_keypoints_sequence()
                    if keypoints_sequence is not None:
                        settings = message.get("settings", {})
                        prediction_result = await process_keypoints_with_model(keypoints_sequence, settings)
                        prediction_result["source"] = "upload"
                        prediction_result["hands_detected"] = True  # Predicción exitosa implica detección
                        prediction_result["timestamp"] = time.time()
                        prediction_result["status"] = f"✅ Predicción completada: {prediction_result.get('main_prediction', 'Error')}"
                        
                        print(f"🔍 DEBUG PREDICCIÓN MANUAL: Enviando prediction_result con source='{prediction_result.get('source')}' y status='{prediction_result.get('status')}'")
                        print(f"🔍 DEBUG PREDICCIÓN MANUAL: hands_detected={prediction_result.get('hands_detected')}")
                        
                        await manager.send_message(websocket, prediction_result)
                        video_upload_processor.reset()
                    else:
                        await manager.send_message(websocket, {
                            "source": "upload",
                            "error": "No se pudieron obtener suficientes keypoints válidos del video"
                        })
                elif len(video_upload_processor.keypoints_buffer) < 30:
                    await manager.send_message(websocket, {
                        "source": "upload",
                        "error": f"Video con muy pocas detecciones de manos: {len(video_upload_processor.keypoints_buffer)}/30 frames mínimos requeridos"
                    })
                else:
                    # Ya se procesó durante la extracción
                    print("🎯 VIDEO UPLOAD: Ya procesado durante extracción")
                
                # IMPORTANTE: Restaurar funcionalidad de cámara al finalizar video upload
                is_processing_video_upload = False
                
                # Cancelar timeout de seguridad
                if video_upload_timeout_task and not video_upload_timeout_task.done():
                    video_upload_timeout_task.cancel()
                
                print("📷 CÁMARA: Restaurada funcionalidad - Video upload completado")
                
                # Notificar al frontend que la cámara está disponible nuevamente
                await manager.send_message(websocket, {
                    "type": "camera_restored",
                    "status": "📷 Cámara restaurada - Video upload completado"
                })
                
    except WebSocketDisconnect:
        # Restaurar estado al desconectar
        is_processing_video_upload = False
        if video_upload_timeout_task and not video_upload_timeout_task.done():
            video_upload_timeout_task.cancel()
        manager.disconnect(websocket)
        logger.info("Cliente WebSocket desconectado")
    except Exception as e:
        # Restaurar estado en caso de error
        is_processing_video_upload = False
        if video_upload_timeout_task and not video_upload_timeout_task.done():
            video_upload_timeout_task.cancel()
        logger.error(f"Error en WebSocket: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor LSP-AYNI...")
    
    # Configuración del servidor desde config
    uvicorn.run(
        "main:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=SERVER_CONFIG["reload"],
        log_level=SERVER_CONFIG["log_level"]
    )
