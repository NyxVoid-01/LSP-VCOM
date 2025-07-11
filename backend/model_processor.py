import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional, Dict, Any
import logging

class ModelPreprocessor:
    """
    Preprocesador de datos para el modelo PUCP-GLOSAS
    Aplica las mismas transformaciones usadas durante el entrenamiento
    """
    
    def __init__(self, target_frames: int = 50):
        self.target_frames = target_frames
        self.logger = logging.getLogger(__name__)
        
    def normalize_sequence_length(self, keypoints: np.ndarray, target_frames: int = None) -> np.ndarray:
        """
        Normaliza la longitud de secuencia usando interpolación lineal
        
        Args:
            keypoints: Array de keypoints de forma (frames, keypoints, coords)
            target_frames: Número objetivo de frames (default: self.target_frames)
            
        Returns:
            Array normalizado de forma (target_frames, keypoints, coords)
        """
        if target_frames is None:
            target_frames = self.target_frames
            
        current_frames = keypoints.shape[0]
        
        if current_frames == target_frames:
            return keypoints
        
        # Interpolación lineal para cada keypoint y coordenada
        old_indices = np.linspace(0, current_frames - 1, current_frames)
        new_indices = np.linspace(0, current_frames - 1, target_frames)
        
        normalized = np.zeros((target_frames, keypoints.shape[1], keypoints.shape[2]))
        
        for kp in range(keypoints.shape[1]):
            for coord in range(keypoints.shape[2]):
                normalized[:, kp, coord] = np.interp(
                    new_indices, 
                    old_indices, 
                    keypoints[:, kp, coord]
                )
        
        return normalized
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normaliza keypoints usando z-score (media=0, std=1)
        
        Args:
            keypoints: Array de keypoints de forma (frames, keypoints, coords)
            
        Returns:
            Array normalizado con la misma forma
        """
        original_shape = keypoints.shape
        flattened = keypoints.reshape(-1, keypoints.shape[-1])
        
        # Calcular estadísticas
        mean = np.mean(flattened, axis=0)
        std = np.std(flattened, axis=0) + 1e-8  # Evitar división por cero
        
        # Normalizar
        normalized = (flattened - mean) / std
        
        return normalized.reshape(original_shape)
    
    def validate_keypoints_shape(self, keypoints: np.ndarray) -> bool:
        """
        Valida que los keypoints tengan la forma correcta
        
        Args:
            keypoints: Array de keypoints
            
        Returns:
            True si la forma es válida
        """
        if len(keypoints.shape) != 3:
            self.logger.error(f"Forma incorrecta: esperado 3D, recibido {len(keypoints.shape)}D")
            return False
            
        frames, kp_count, coords = keypoints.shape
        
        if kp_count != 42:
            self.logger.error(f"Número incorrecto de keypoints: esperado 42, recibido {kp_count}")
            return False
            
        if coords != 2:
            self.logger.error(f"Número incorrecto de coordenadas: esperado 2, recibido {coords}")
            return False
            
        if frames < 21:
            self.logger.warning(f"Secuencia incompleta: {frames} frames (se requieren al menos 21)")
            
        return True
    
    def preprocess_sequence(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Aplica todo el pipeline de preprocesamiento
        
        Args:
            keypoints: Array de keypoints raw de forma (frames, 42, 2)
            
        Returns:
            Array preprocesado de forma (1, target_frames, 42, 2) listo para el modelo
            None si hay error en el procesamiento
        """
        try:
            # Validar entrada
            if not self.validate_keypoints_shape(keypoints):
                return None
            
            self.logger.info(f"Preprocesando secuencia: {keypoints.shape}")
            
            # 1. Normalizar longitud de secuencia
            normalized_length = self.normalize_sequence_length(keypoints, self.target_frames)
            self.logger.debug(f"Después de normalizar longitud: {normalized_length.shape}")
            
            # 2. Normalizar valores (z-score)
            normalized_values = self.normalize_keypoints(normalized_length)
            self.logger.debug(f"Después de normalizar valores: {normalized_values.shape}")
            
            # 3. Añadir dimensión de batch
            batch_ready = np.expand_dims(normalized_values, axis=0)
            self.logger.debug(f"Listo para modelo: {batch_ready.shape}")
            
            return batch_ready
            
        except Exception as e:
            self.logger.error(f"Error en preprocesamiento: {e}")
            return None
    
    def check_data_quality(self, keypoints: np.ndarray) -> Dict[str, Any]:
        """
        Analiza la calidad de los datos de keypoints
        
        Args:
            keypoints: Array de keypoints
            
        Returns:
            Diccionario con métricas de calidad
        """
        quality_metrics = {
            'valid_shape': self.validate_keypoints_shape(keypoints),
            'frames_count': keypoints.shape[0] if len(keypoints.shape) >= 1 else 0,
            'has_nan': np.isnan(keypoints).any(),
            'has_inf': np.isinf(keypoints).any(),
            'coordinate_range': {
                'min': float(np.min(keypoints)),
                'max': float(np.max(keypoints)),
                'mean': float(np.mean(keypoints)),
                'std': float(np.std(keypoints))
            }
        }
        
        # Verificar movimiento (varianza en el tiempo)
        if len(keypoints.shape) == 3 and keypoints.shape[0] > 1:
            frame_variance = np.var(keypoints, axis=0)
            quality_metrics['movement_variance'] = {
                'mean': float(np.mean(frame_variance)),
                'min': float(np.min(frame_variance)),
                'max': float(np.max(frame_variance))
            }
        
        # Verificar si las coordenadas están en rango esperado (0-1 para MediaPipe)
        in_range = np.all((keypoints >= 0) & (keypoints <= 1))
        quality_metrics['coordinates_in_range'] = bool(in_range)
        
        return quality_metrics


class SignLanguageModel:
    """
    Wrapper para el modelo de reconocimiento de señas PUCP-GLOSAS
    """
    
    def __init__(self, model_path: str, encoder_path: str, info_path: str):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.info_path = Path(info_path)
        
        self.model = None
        self.label_encoder = None
        self.model_info = None
        self.preprocessor = ModelPreprocessor()
        
        self.logger = logging.getLogger(__name__)
        
    def load_model_components(self) -> bool:
        """
        Carga el modelo, encoder y metadatos
        
        Returns:
            True si se cargó exitosamente
        """
        try:
            # Cargar modelo
            self.logger.info(f"Cargando modelo desde: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info("✅ Modelo cargado exitosamente")
            
            # Cargar label encoder
            self.logger.info(f"Cargando label encoder desde: {self.encoder_path}")
            with open(self.encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.logger.info("✅ Label encoder cargado exitosamente")
            
            # Cargar información del modelo
            self.logger.info(f"Cargando info del modelo desde: {self.info_path}")
            with open(self.info_path, 'rb') as f:
                self.model_info = pickle.load(f)
            self.logger.info("✅ Info del modelo cargada exitosamente")
            
            # Verificar consistencia
            if self.model_info:
                expected_classes = self.model_info.get('num_classes', 0)
                actual_classes = len(self.label_encoder.classes_)
                
                if expected_classes != actual_classes:
                    self.logger.warning(
                        f"Inconsistencia en número de clases: "
                        f"esperado {expected_classes}, encontrado {actual_classes}"
                    )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando componentes del modelo: {e}")
            return False
    
    def predict(self, keypoints: np.ndarray, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """
        Realiza predicción sobre keypoints
        
        Args:
            keypoints: Array de keypoints raw (frames, 42, 2)
            top_k: Número de predicciones top a retornar
            
        Returns:
            Diccionario con resultados de predicción o None si hay error
        """
        try:
            if self.model is None or self.label_encoder is None:
                self.logger.error("Modelo no cargado. Llama load_model_components() primero")
                return None
            
            # Preprocesar datos
            processed_data = self.preprocessor.preprocess_sequence(keypoints)
            if processed_data is None:
                self.logger.error("Error en preprocesamiento de datos")
                return None
            
            self.logger.info(f"Realizando predicción sobre datos de forma: {processed_data.shape}")
            
            # Realizar predicción
            predictions = self.model.predict(processed_data, verbose=0)
            
            # Obtener probabilidades y clases
            probabilities = predictions[0]  # Remover dimensión de batch
            
            # Obtener top K predicciones
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            
            results = {
                'predictions': [],
                'main_prediction': None,
                'confidence': 0.0,
                'raw_probabilities': probabilities.tolist(),
                'processing_info': {
                    'input_shape': keypoints.shape,
                    'processed_shape': processed_data.shape,
                    'model_input_shape': self.model.input_shape if self.model else None
                }
            }
            
            # Procesar predicciones top-k
            for i, idx in enumerate(top_indices):
                class_name = self.label_encoder.classes_[idx]
                confidence = float(probabilities[idx])
                
                prediction = {
                    'rank': i + 1,
                    'label': class_name,
                    'confidence': confidence,
                    'class_index': int(idx)
                }
                
                results['predictions'].append(prediction)
                
                # La primera es la predicción principal
                if i == 0:
                    results['main_prediction'] = class_name
                    results['confidence'] = confidence
            
            self.logger.info(f"Predicción exitosa: {results['main_prediction']} ({results['confidence']:.3f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información del modelo
        
        Returns:
            Diccionario con información del modelo
        """
        if self.model_info:
            return self.model_info.copy()
        
        # Información básica si no hay archivo de info
        return {
            'num_classes': len(self.label_encoder.classes_) if self.label_encoder else 0,
            'class_names': self.label_encoder.classes_.tolist() if self.label_encoder else [],
            'model_loaded': self.model is not None,
            'input_shape': self.model.input_shape if self.model else None
        }
    
    def is_ready(self) -> bool:
        """
        Verifica si el modelo está listo para predicciones
        
        Returns:
            True si está listo
        """
        return (self.model is not None and 
                self.label_encoder is not None and 
                self.preprocessor is not None)


# Función de utilidad para testing
def test_preprocessor():
    """Función de prueba para el preprocesador"""
    print("🧪 Testing ModelPreprocessor...")
    
    preprocessor = ModelPreprocessor(target_frames=50)
    
    # Crear datos de prueba
    test_keypoints = np.random.random((30, 42, 2))  # 30 frames, 42 keypoints, coordenadas x,y
    print(f"📊 Datos de prueba: {test_keypoints.shape}")
    
    # Test preprocesamiento
    processed = preprocessor.preprocess_sequence(test_keypoints)
    if processed is not None:
        print(f"✅ Preprocesamiento exitoso: {processed.shape}")
    else:
        print("❌ Error en preprocesamiento")
    
    # Test calidad de datos
    quality = preprocessor.check_data_quality(test_keypoints)
    print(f"📈 Calidad de datos: {quality}")

if __name__ == "__main__":
    test_preprocessor()
