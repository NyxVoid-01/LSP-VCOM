import cv2
import mediapipe as mp
import numpy as np
import base64
from PIL import Image
import io
import time
from typing import Optional, Tuple

class HandKeypointExtractor:
    """
    Extractor de keypoints de manos usando MediaPipe
    Extrae 42 keypoints (21 por cada mano) en tiempo real
    """
    
    def __init__(self):
        # Configuración de MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configurar detector de manos optimizado para fondos complejos y simples
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Máximo 2 manos
            min_detection_confidence=0.5,  # Balance entre robustez y precisión
            min_tracking_confidence=0.5,   # Tracking estable sin ser demasiado estricto
            model_complexity=1             # Modelo de complejidad media para mejor rendimiento
        )
        
        # Estado del extractor
        self.is_recording = False
        self.countdown_active = False
        self.is_paused = False  # Nueva pausa después de predicción
        self.keypoints_buffer = []
        self.recording_start_time = None
        self.pause_start_time = None
        self.recording_duration = 2.8  # 2.8 segundos - buffer para asegurar 50 frames
        self.countdown_duration = 3.0  # 3 segundos de cuenta regresiva
        self.pause_duration = 2.0  # 2 segundos de pausa después de predicción
        
        # Contador simple para estabilidad de detección
        self.consecutive_good_frames = 0
        
        # Variable para tracking del countdown
        self.countdown_remaining = 0
        
        # Estado anterior para evitar log spam
        self.previous_log_state = None
        
        # Modo de procesamiento (camera o upload)
        self.processing_mode = "camera"  # Default: camera
        
        # Configuración de resize optimizado
        self.target_width = 640
        self.target_height = 480
        
    def _resize_frame_optimized(self, frame: np.ndarray) -> np.ndarray:
        """
        Redimensiona el frame a 640x480 manteniendo aspect ratio
        y añadiendo padding negro si es necesario
        """
        h, w = frame.shape[:2]
        
        # Si ya es 640x480, no hacer nada
        if w == self.target_width and h == self.target_height:
            return frame
        
        # Calcular aspect ratio y dimensiones de resize
        aspect_ratio = w / h
        target_aspect = self.target_width / self.target_height
        
        if aspect_ratio > target_aspect:
            # Frame más ancho - ajustar por ancho
            new_width = self.target_width
            new_height = int(self.target_width / aspect_ratio)
        else:
            # Frame más alto - ajustar por alto
            new_height = self.target_height
            new_width = int(self.target_height * aspect_ratio)
        
        # Resize manteniendo aspect ratio
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Crear frame de destino con fondo negro
        result_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Calcular posición para centrar
        y_offset = (self.target_height - new_height) // 2
        x_offset = (self.target_width - new_width) // 2
        
        # Colocar frame redimensionado en el centro
        result_frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame
        
        return result_frame

    def detect_hands_in_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Detecta manos y extrae 42 keypoints (21 por cada mano)
        
        Args:
            frame: Frame de video (numpy array)
            
        Returns:
            Tuple (hands_detected, keypoints, status)
            - hands_detected: True si se detectan exactamente 2 manos estables
            - keypoints: Array de keypoints (42, 2) o None
            - status: String indicando el estado de detección
        """
        # Redimensionar frame a 640x480 de forma optimizada
        optimized_frame = self._resize_frame_optimized(frame)
        
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(optimized_frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame con MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Crear copia del frame optimizado para anotaciones
        annotated_frame = optimized_frame.copy()
        
        # Debug: imprimir información de detección (solo cambios importantes)
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        # Solo log durante detección inicial para modo camera
        if (num_hands > 0 and not self.countdown_active and not self.is_recording and not self.is_paused 
            and self.processing_mode == "camera"):
            current_state = f"hands:{num_hands}"
            if self.previous_log_state != current_state:
                print(f"🔍 DEBUG: Detectadas {num_hands} manos")
                self.previous_log_state = current_state
        
        # Verificar si se detectaron EXACTAMENTE 2 manos para 42 keypoints
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            
            # Extraer keypoints de ambas manos
            keypoints = self._extract_keypoints(results)
            
            if keypoints is not None:
                # Solo log durante detección inicial para modo camera
                if (not self.countdown_active and not self.is_recording and not self.is_paused 
                    and self.processing_mode == "camera"):
                    keypoints_state = f"keypoints:extracted"
                    if self.previous_log_state != keypoints_state:
                        print(f"✅ 42 KEYPOINTS EXTRAÍDOS: {keypoints.shape}")
                        self.previous_log_state = keypoints_state
                
                # Dibujar anotaciones básicas
                self._draw_annotations(annotated_frame, results)
                
                # Incrementar contador de frames consecutivos buenos
                self.consecutive_good_frames += 1
                
                # TEMPORAL: Reducir a 1 frame para debug
                if self.consecutive_good_frames >= 1:
                    # Solo log la primera vez que se estabilizan las manos para modo camera
                    if (not self.countdown_active and not self.is_recording and not self.is_paused 
                        and self.processing_mode == "camera"):
                        stable_state = "hands:stable"
                        if self.previous_log_state != stable_state:
                            print(f"🎯 AMBAS MANOS ESTABLES - INICIANDO FLUJO")
                            self.previous_log_state = stable_state
                    return True, keypoints, "hands_detected"
                else:
                    # Solo log durante detección inicial para modo camera
                    if (not self.countdown_active and not self.is_recording and not self.is_paused 
                        and self.processing_mode == "camera"):
                        waiting_state = f"waiting:{self.consecutive_good_frames}"
                        if self.previous_log_state != waiting_state:
                            print(f"⏳ Esperando más frames consecutivos: {self.consecutive_good_frames}/1")
                            self.previous_log_state = waiting_state
            else:
                pass  # Error extrayendo keypoints - silencioso
        else:
            pass  # Se requieren 2 manos - silencioso
        
        # Reset contador si no hay detección buena
        if self.consecutive_good_frames > 0:
            self.consecutive_good_frames = 0
        return False, None, "no_hands"
    
    def _extract_keypoints(self, results) -> Optional[np.ndarray]:
        """
        Extrae 42 keypoints de ambas manos detectadas
        """
        try:
            keypoints = np.zeros((42, 2), dtype=np.float32)
            
            # Organizar manos por posición (izquierda/derecha en la imagen)
            hands_data = []
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    x = landmark.x  # Normalizado (0-1)
                    y = landmark.y  # Normalizado (0-1)
                    hand_keypoints.append([x, y])
                hands_data.append((i, hand_keypoints))
            
            # Ordenar por posición X (izquierda a derecha)
            hands_data.sort(key=lambda x: np.mean([kp[0] for kp in x[1]]))
            
            # Llenar keypoints: primera mano (izq) 0-20, segunda mano (der) 21-41
            for hand_idx, (_, hand_keypoints) in enumerate(hands_data):
                if hand_idx < 2:  # Solo primeras 2 manos
                    start_idx = hand_idx * 21
                    end_idx = start_idx + 21
                    keypoints[start_idx:end_idx] = np.array(hand_keypoints)
            
            return keypoints
            
        except Exception as e:
            # Error silencioso para no llenar logs
            return None
    
    def _draw_annotations(self, frame, results):
        """
        Dibuja anotaciones de manos en el frame
        """
        try:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        except Exception as e:
            # Error silencioso
            pass
    
    async def start_countdown(self) -> bool:
        """
        Inicia cuenta regresiva de 3 segundos
        """
        if self.countdown_active or self.is_recording:
            return False
            
        self.countdown_active = True
        self.countdown_remaining = int(self.countdown_duration)
        
        # Cuenta regresiva
        import asyncio
        for i in range(int(self.countdown_duration), 0, -1):
            self.countdown_remaining = i
            # Solo log el countdown una vez por segundo
            countdown_state = f"countdown:{i}"
            if self.previous_log_state != countdown_state:
                print(f"🔥 Iniciando grabación en {i} segundos...")
                self.previous_log_state = countdown_state
            await asyncio.sleep(1.0)
            
            if not self.countdown_active:  # Por si se cancela
                return False
        
        print("🎬 ¡GRABANDO!")
        self.countdown_active = False
        self.countdown_remaining = 0
        return True
    
    def start_recording(self):
        """Inicia la grabación de keypoints"""
        if self.is_recording:
            return False
            
        self.is_recording = True
        self.keypoints_buffer = []
        self.recording_start_time = time.time()
        # Solo log una vez al iniciar
        recording_state = "recording:started"
        if self.previous_log_state != recording_state:
            print(f"📹 Iniciando grabación por {self.recording_duration} segundos...")
            self.previous_log_state = recording_state
        return True
    
    def add_keypoints_to_buffer(self, keypoints: np.ndarray):
        """Añade keypoints al buffer durante la grabación"""
        if self.is_recording and keypoints is not None:
            self.keypoints_buffer.append(keypoints.copy())
    
    def should_stop_recording(self) -> bool:
        """Verifica si debe terminar la grabación"""
        if not self.is_recording:
            return False
            
        elapsed_time = time.time() - self.recording_start_time
        return elapsed_time >= self.recording_duration
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """
        Detiene la grabación y retorna los keypoints capturados
        """
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        if len(self.keypoints_buffer) == 0:
            # No se capturaron keypoints - silencioso
            return None
        
        # Convertir buffer a numpy array
        keypoints_sequence = np.array(self.keypoints_buffer)
        # Log silencioso - el main.py ya maneja este log
        # print(f"✅ Grabación terminada: {len(self.keypoints_buffer)} frames capturados")
        
        # Limpiar buffer
        self.keypoints_buffer = []
        
        return keypoints_sequence
    
    def cancel_recording(self):
        """Cancela la grabación actual"""
        self.is_recording = False
        self.countdown_active = False
        self.is_paused = False
        self.keypoints_buffer = []
        self.previous_log_state = None  # Reset state tracking
        # print("❌ Grabación cancelada")  # Comentado
    
    def start_pause(self):
        """Inicia pausa después de una predicción"""
        if self.is_paused:
            return False
        
        # Cancelar cualquier countdown o grabación en curso
        if self.countdown_active or self.is_recording:
            self.countdown_active = False
            self.is_recording = False
            
        self.is_paused = True
        self.pause_start_time = time.time()
        self.previous_log_state = None  # Reset state tracking para evitar logs durante pausa
        print(f"⏸️ INICIANDO PAUSA de {self.pause_duration} segundos después de predicción")
        return True
    
    def should_end_pause(self) -> bool:
        """Verifica si debe terminar la pausa"""
        if not self.is_paused:
            return False
            
        elapsed_time = time.time() - self.pause_start_time
        return elapsed_time >= self.pause_duration
    
    def end_pause(self):
        """Termina la pausa y vuelve al estado normal"""
        if self.is_paused:
            self.is_paused = False
            self.pause_start_time = None
            self.previous_log_state = None  # Reset state tracking al terminar pausa
            print("▶️ PAUSA TERMINADA - Listo para nueva detección")
    
    def get_recording_progress(self) -> float:
        """Retorna el progreso de la grabación (0.0 a 1.0)"""
        if not self.is_recording:
            return 0.0
            
        elapsed_time = time.time() - self.recording_start_time
        progress = min(elapsed_time / self.recording_duration, 1.0)
        return progress
    
    def process_base64_frame(self, base64_data: str) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Procesa un frame codificado en base64
        
        Returns:
            Tuple (hands_detected, keypoints, status_message)
        """
        try:
            # Decodificar base64
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detectar manos
            hands_detected, keypoints, detection_status = self.detect_hands_in_frame(frame)
            
            # Verificar si estamos en pausa
            if self.is_paused:
                if self.should_end_pause():
                    self.end_pause()
                    # Reset state tracking al terminar pausa
                    self.previous_log_state = None
                    return hands_detected, keypoints, "pause_ended"
                else:
                    remaining_time = self.pause_duration - (time.time() - self.pause_start_time)
                    return hands_detected, keypoints, f"paused:{remaining_time:.1f}"
            
            if hands_detected:
                if self.countdown_active:
                    return True, keypoints, f"countdown:{self.countdown_remaining}"
                elif self.is_recording:
                    self.add_keypoints_to_buffer(keypoints)
                    progress = self.get_recording_progress()
                    return True, keypoints, f"recording:{progress:.2f}"
                else:
                    # Reset state tracking cuando volvemos a detección normal
                    self.previous_log_state = None
                    return True, keypoints, detection_status
            else:
                # Si estamos grabando pero no detectamos manos, seguir grabando
                if self.is_recording:
                    progress = self.get_recording_progress()
                    return False, None, f"recording_no_hands:{progress:.2f}"
                else:
                    # Usar el status de detección directamente
                    return False, None, detection_status
                    
        except Exception as e:
            # Error silencioso
            return False, None, f"error:{str(e)}"
    
    def get_status_message(self, hands_detected: bool) -> str:
        """Genera mensaje de estado basado en el estado actual"""
        if self.is_paused:
            remaining_time = max(0, self.pause_duration - (time.time() - self.pause_start_time))
            return f"⏸️ Pausa después de predicción... {remaining_time:.1f}s"
        elif self.countdown_active:
            return "🔥 Preparándose para grabar... ¡Mantén tus manos visibles!"
        elif self.is_recording:
            progress = self.get_recording_progress()
            progress_percent = int(progress * 100)
            return f"🎬 Grabando... {progress_percent}% completado"
        elif hands_detected:
            return "✋ Ambas manos detectadas - Listo para grabar"
        else:
            return "👋 Muestra ambas manos frente a la cámara"
    
    def cleanup(self):
        """Limpia recursos del extractor"""
        if self.hands:
            self.hands.close()
        self.cancel_recording()