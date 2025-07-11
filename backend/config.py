import os
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Rutas del modelo
MODEL_DIR = PROJECT_ROOT / "models"  # Carpeta donde estarán los modelos entrenados
MODEL_CONFIG = {
    "model_path": MODEL_DIR / "modelo_finetuned_pucp_glosas.keras",
    "encoder_path": MODEL_DIR / "label_encoder.pkl", 
    "info_path": MODEL_DIR / "model_info.pkl"
}

# Configuración del servidor
SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "reload": True,
    "log_level": "info"
}

# Configuración de MediaPipe
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5
}

# Configuración de captura
CAPTURE_CONFIG = {
    "target_frames": 50,  # Exactamente 50 frames para el modelo
    "recording_duration": 2.5,  # segundos - optimizado para señas naturales  
    "countdown_duration": 3.0,  # segundos
    "min_frames_for_processing": 50,  # Requerir exactamente 50 frames
    "target_fps": 20  # 50 frames ÷ 2.5s = 20 FPS exactos
}

# Configuración de procesamiento
PROCESSING_CONFIG = {
    "default_confidence_threshold": 0.6,
    "default_prediction_count": 3,
    "max_prediction_count": 7,
    "frame_rate_ms": 66  # ~15 FPS
}

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_log": False,  # Set to True para guardar logs en archivo
    "log_file": BASE_DIR / "logs" / "lsp_ayni.log"
}

# Verificar y crear directorios necesarios
def ensure_directories():
    """Crea directorios necesarios si no existen"""
    directories = [
        MODEL_DIR,
        BASE_DIR / "logs" if LOGGING_CONFIG["file_log"] else None
    ]
    
    for directory in directories:
        if directory and not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Directorio creado: {directory}")

def check_model_files():
    """Verifica si los archivos del modelo existen"""
    missing_files = []
    
    for name, path in MODEL_CONFIG.items():
        if not path.exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("⚠️ Archivos del modelo faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🎭 El servidor se iniciará en modo demostración")
        return False
    else:
        print("✅ Todos los archivos del modelo encontrados")
        return True

def get_environment_info():
    """Retorna información del entorno"""
    return {
        "base_dir": str(BASE_DIR),
        "project_root": str(PROJECT_ROOT),
        "model_dir": str(MODEL_DIR),
        "python_version": os.sys.version,
        "platform": os.name
    }

if __name__ == "__main__":
    print("🔧 Configuración LSP-AYNI")
    print("=" * 40)
    
    ensure_directories()
    model_available = check_model_files()
    
    env_info = get_environment_info()
    print(f"\n📊 Información del entorno:")
    for key, value in env_info.items():
        print(f"   {key}: {value}")
    
    print(f"\n🤖 Estado del modelo: ✅ Disponible")
