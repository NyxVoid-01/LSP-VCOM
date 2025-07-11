#!/usr/bin/env python3
"""
Utilidades para el servidor LSP-AYNI
Script para facilitar el inicio, testing y mantenimiento del servidor
"""

import asyncio
import argparse
import sys
import os
import subprocess
from pathlib import Path
import json
import time

# Agregar directorio backend al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    SERVER_CONFIG, MODEL_CONFIG, CAPTURE_CONFIG, 
    ensure_directories, check_model_files, get_environment_info
)

def install_dependencies():
    """Instala las dependencias del proyecto"""
    print("📦 Instalando dependencias...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Archivo requirements.txt no encontrado")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...")
    
    # Mapeo de nombres de paquetes pip a nombres de módulos de Python
    package_mappings = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn", 
        "opencv-python": "cv2",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "tensorflow": "tensorflow",
        "pillow": "PIL",
        "scikit-learn": "sklearn"
    }
    
    missing_packages = []
    
    for package_name, module_name in package_mappings.items():
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ Paquetes faltantes: {', '.join(missing_packages)}")
        print("💡 Ejecuta: python utils.py --install-deps")
        return False
    else:
        print("✅ Todas las dependencias están instaladas")
        return True

def test_camera():
    """Prueba el acceso a la cámara"""
    print("📹 Probando acceso a la cámara...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ No se pudo acceder a la cámara")
            return False
        
        ret, frame = cap.read()
        
        if ret:
            print(f"✅ Cámara funcionando - Resolución: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print("❌ No se pudo capturar frame de la cámara")
            return False
        
        cap.release()
        return True
        
    except ImportError:
        print("❌ OpenCV no está instalado")
        return False
    except Exception as e:
        print(f"❌ Error probando cámara: {e}")
        return False

def test_mediapipe():
    """Prueba la funcionalidad de MediaPipe"""
    print("👋 Probando MediaPipe...")
    
    try:
        import mediapipe as mp
        import numpy as np
        
        # Crear datos de prueba
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Inicializar MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # Procesar imagen de prueba
        results = hands.process(test_image)
        
        hands.close()
        print("✅ MediaPipe funcionando correctamente")
        return True
        
    except ImportError:
        print("❌ MediaPipe no está instalado")
        return False
    except Exception as e:
        print(f"❌ Error probando MediaPipe: {e}")
        return False

def test_tensorflow():
    """Prueba TensorFlow"""
    print("🧠 Probando TensorFlow...")
    
    try:
        import tensorflow as tf
        
        print(f"   📊 Versión: {tf.__version__}")
        print(f"   💻 GPU disponible: {tf.config.list_physical_devices('GPU')}")
        
        # Test simple
        tensor = tf.constant([1, 2, 3, 4])
        print(f"   ✅ Test tensor: {tensor}")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow no está instalado")
        return False
    except Exception as e:
        print(f"❌ Error probando TensorFlow: {e}")
        return False

def test_keypoint_extractor():
    """Prueba el extractor de keypoints"""
    print("🔍 Probando extractor de keypoints...")
    
    try:
        from keypoint_extractor import HandKeypointExtractor
        
        extractor = HandKeypointExtractor()
        print("✅ Extractor inicializado")
        
        # Test con datos simulados
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        hands_detected, keypoints, annotated_frame = extractor.detect_hands_in_frame(test_frame)
        print(f"   📊 Manos detectadas: {hands_detected}")
        
        extractor.cleanup()
        print("✅ Extractor de keypoints funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Error probando extractor: {e}")
        return False

def test_model_processor():
    """Prueba el procesador del modelo"""
    print("⚙️ Probando procesador del modelo...")
    
    try:
        from model_processor import ModelPreprocessor
        
        preprocessor = ModelPreprocessor(target_frames=50)
        
        # Test con datos simulados
        import numpy as np
        test_keypoints = np.random.random((30, 42, 2))
        
        processed = preprocessor.preprocess_sequence(test_keypoints)
        
        if processed is not None:
            print(f"   ✅ Preprocesamiento exitoso: {processed.shape}")
        else:
            print("   ❌ Error en preprocesamiento")
            return False
        
        quality = preprocessor.check_data_quality(test_keypoints)
        print(f"   📈 Análisis de calidad completado")
        
        print("✅ Procesador del modelo funcionando")
        return True
        
    except Exception as e:
        print(f"❌ Error probando procesador: {e}")
        return False

def run_full_test():
    """Ejecuta todas las pruebas del sistema"""
    print("🧪 EJECUTANDO PRUEBAS COMPLETAS DEL SISTEMA")
    print("=" * 50)
    
    tests = [
        ("Dependencias", check_dependencies),
        ("Cámara", test_camera),
        ("MediaPipe", test_mediapipe),
        ("TensorFlow", test_tensorflow),
        ("Extractor de Keypoints", test_keypoint_extractor),
        ("Procesador del Modelo", test_model_processor)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Error crítico en {test_name}: {e}")
            results[test_name] = False
    
    print(f"\n📊 RESUMEN DE PRUEBAS")
    print("=" * 30)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todos los componentes funcionando correctamente!")
        print("💡 Puedes iniciar el servidor con: python utils.py --start")
    else:
        print("⚠️ Algunos componentes tienen problemas")
        print("💡 Revisa los errores e instala dependencias faltantes")
    
    return passed == total

def start_server():
    """Inicia el servidor"""
    print("🚀 Iniciando servidor LSP-AYNI...")
    
    # Verificar configuración
    ensure_directories()
    model_available = check_model_files()
    
    if not model_available:
        print("🎭 Iniciando en modo demostración")
    
    # Importar y ejecutar servidor
    try:
        import uvicorn
        from main import app
        
        print(f"🌐 Servidor disponible en: http://{SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
        print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
        print("⏹️ Presiona Ctrl+C para detener")
        
        uvicorn.run(
            app,
            host=SERVER_CONFIG['host'],
            port=SERVER_CONFIG['port'],
            reload=SERVER_CONFIG['reload'],
            log_level=SERVER_CONFIG['log_level']
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        print(f"❌ Error iniciando servidor: {e}")

def show_info():
    """Muestra información del sistema"""
    print("ℹ️ INFORMACIÓN DEL SISTEMA LSP-AYNI")
    print("=" * 40)
    
    # Información del entorno
    env_info = get_environment_info()
    print("\n📊 Entorno:")
    for key, value in env_info.items():
        print(f"   {key}: {value}")
    
    # Estado del modelo
    print(f"\n🤖 Estado del modelo:")
    model_available = check_model_files()
    
    # Configuración
    print(f"\n⚙️ Configuración:")
    print(f"   Servidor: {SERVER_CONFIG['host']}:{SERVER_CONFIG['port']}")
    print(f"   Frames objetivo: {CAPTURE_CONFIG['target_frames']}")
    print(f"   Duración grabación: {CAPTURE_CONFIG['recording_duration']}s")
    print(f"   Duración countdown: {CAPTURE_CONFIG['countdown_duration']}s")

def main():
    """Función principal del script de utilidades"""
    parser = argparse.ArgumentParser(
        description="Utilidades para el servidor LSP-AYNI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python utils.py --install-deps    # Instalar dependencias
  python utils.py --test           # Ejecutar todas las pruebas
  python utils.py --start          # Iniciar el servidor
  python utils.py --info           # Mostrar información del sistema
        """
    )
    
    parser.add_argument('--install-deps', action='store_true',
                       help='Instalar dependencias desde requirements.txt')
    parser.add_argument('--test', action='store_true',
                       help='Ejecutar todas las pruebas del sistema')
    parser.add_argument('--start', action='store_true',
                       help='Iniciar el servidor')
    parser.add_argument('--info', action='store_true',
                       help='Mostrar información del sistema')
    parser.add_argument('--check-camera', action='store_true',
                       help='Probar acceso a la cámara')
    parser.add_argument('--check-deps', action='store_true',
                       help='Verificar dependencias')
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
    elif args.test:
        run_full_test()
    elif args.start:
        start_server()
    elif args.info:
        show_info()
    elif args.check_camera:
        test_camera()
    elif args.check_deps:
        check_dependencies()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
