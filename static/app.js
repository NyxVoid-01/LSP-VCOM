// ===== GLOBAL VARIABLES =====
let video = document.getElementById('video');
let canvas = document.createElement('canvas');
let ctx = canvas.getContext('2d');
let stream = null;
let ws = null;
let isStreaming = false;
let reconnectAttempts = 0;
let sessionStartTime = null;
let predictionCount = 0;
let predictionHistory = [];
let isProcessingVideoUpload = false;
let guidanceBlockInterval = null;  // Interval para forzar ocultamiento durante video upload
let currentSettings = {
    confidenceThreshold: 0.6,
    predictionCount: 3,
    frameRate: 50  // 20 FPS (1000ms ÷ 20 = 50ms) para capturar exactamente 50 frames en 2.5s
};

const maxReconnectAttempts = 5;

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    updateConnectionStatus('disconnected');
});

function initializeApp() {
    console.log('🚀 AI SignLang - Sistema de Reconocimiento de Lenguaje de Señas');
    console.log('📡 Servidor:', window.location.host);
    
    // Initialize settings from localStorage
    loadSettings();
    updateSettingsUI();
    
    // ASEGURAR que countdown esté oculto al inicio
    hideCountdown();
    
    // Start session timer
    sessionStartTime = Date.now();
    updateSessionTimer();
    setInterval(updateSessionTimer, 1000);
}

function setupEventListeners() {
    // Settings controls
    const confidenceSlider = document.getElementById('confidenceThreshold');
    const confidenceValue = document.getElementById('confidenceValue');
    
    if (confidenceSlider && confidenceValue) {
        confidenceSlider.addEventListener('input', function() {
            currentSettings.confidenceThreshold = parseFloat(this.value);
            confidenceValue.textContent = this.value;
            saveSettings();
        });
    }
    
    const predictionSelect = document.getElementById('predictionCount');
    if (predictionSelect) {
        predictionSelect.addEventListener('change', function() {
            currentSettings.predictionCount = parseInt(this.value);
            saveSettings();
        });
    }
    
    const frameRateSelect = document.getElementById('frameRate');
    if (frameRateSelect) {
        frameRateSelect.addEventListener('change', function() {
            currentSettings.frameRate = parseInt(this.value);
            saveSettings();
        });
    }
    
    // Filter buttons
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            filterButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            filterPredictions(this.dataset.filter);
        });
    });
    
    // Fullscreen toggle
    const fullscreenBtn = document.getElementById('toggleFullscreen');
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', toggleFullscreen);
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Window events
    window.addEventListener('beforeunload', cleanup);
    window.addEventListener('focus', handleWindowFocus);
    window.addEventListener('blur', handleWindowBlur);
}

// ===== CAMERA MANAGEMENT =====
async function startCamera() {
    try {
        showLoadingOverlay('Solicitando acceso a la cámara...');
        updateStatus('🔄 Solicitando acceso a la cámara...', 'warning');
        
        const constraints = {
            video: {
                width: { ideal: 1280, max: 1920 },
                height: { ideal: 720, max: 1080 },
                facingMode: 'user',
                frameRate: { ideal: 30, max: 60 }
            }
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        video.onloadedmetadata = function() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            hideVideoOverlay();
            
            // Show initial guidance message
            showGuidanceMessage('Muestra ambas manos frente a la cámara');
            
            connectWebSocket();
            
            updateButtonStates(true);
            updateStatus('📹 Cámara iniciada - Conectando al servidor...', 'success');
            hideLoadingOverlay();
        };
        
    } catch (err) {
        console.error('Error accessing camera:', err);
        handleCameraError(err);
        hideLoadingOverlay();
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (ws) {
        ws.close();
        ws = null;
    }
    
    video.srcObject = null;
    isStreaming = false;
    reconnectAttempts = 0;
    
    showVideoOverlay();
    updateButtonStates(false);
    updateStatus('🛑 Cámara detenida', 'warning');
    updateConnectionStatus('disconnected');
    
    // Hide all visual overlays except predictions
    hideGuidanceMessage();
    hideCountdown();
    hideRecordingIcon();
    hidePauseOverlay();
    
    // Keep predictions visible - don't clear them or hide current prediction
    // hideCurrentPredictionDisplay();
    // clearPredictionsDisplay();
    // hidePredictionSubtitle();
}

function handleCameraError(err) {
    let errorMsg = 'Error accediendo a la cámara: ';
    
    switch(err.name) {
        case 'NotAllowedError':
            errorMsg += 'Permiso denegado. Por favor, permite el acceso a la cámara y recarga la página.';
            break;
        case 'NotFoundError':
            errorMsg += 'No se encontró ninguna cámara en este dispositivo.';
            break;
        case 'NotReadableError':
            errorMsg += 'La cámara está siendo utilizada por otra aplicación.';
            break;
        case 'OverconstrainedError':
            errorMsg += 'No se pudo configurar la cámara con los parámetros solicitados.';
            break;
        default:
            errorMsg += err.message;
    }
    
    updateStatus('❌ ' + errorMsg, 'error');
    showErrorModal('Error de Cámara', errorMsg);
}

// ===== WEBSOCKET MANAGEMENT =====
function connectWebSocketForVideoUpload() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log('📡 VIDEO UPLOAD: Conectando al servidor...');
    // NO actualizar status ni connection status para no interferir con UI de cámara
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        console.log('✅ VIDEO UPLOAD: Conectado al servidor WebSocket');
        // NO activar isStreaming ni startStreaming para no interferir con cámara
        reconnectAttempts = 0;
    };
    
    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (err) {
            console.error('VIDEO UPLOAD: Error parsing WebSocket message:', err);
        }
    };
    
    ws.onclose = function(event) {
        console.log('📡 VIDEO UPLOAD: Conexión WebSocket cerrada');
        // NO cambiar isStreaming ni updateConnectionStatus para no afectar cámara
        
        if (!event.wasClean) {
            console.log('📡 VIDEO UPLOAD: Conexión perdida durante video upload');
            // NO intentar reconectar automáticamente durante video upload
        }
    };
    
    ws.onerror = function(error) {
        console.error('VIDEO UPLOAD: WebSocket error:', error);
        // NO actualizar UI status para no interferir con cámara
    };
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    updateStatus('🔌 Conectando al servidor...', 'warning');
    updateConnectionStatus('connecting');
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = function() {
        updateStatus('✅ Conectado - Realizando predicciones...', 'success');
        updateConnectionStatus('connected');
        isStreaming = true;
        reconnectAttempts = 0;
        startStreaming();
    };
    
    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (err) {
            console.error('Error parsing WebSocket message:', err);
        }
    };
    
    ws.onclose = function(event) {
        isStreaming = false;
        updateConnectionStatus('disconnected');
        
        if (event.wasClean) {
            updateStatus('🔌 Conexión cerrada', 'warning');
        } else {
            updateStatus('❌ Conexión perdida - Intentando reconectar...', 'error');
            attemptReconnect();
        }
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        updateStatus('❌ Error de conexión', 'error');
        updateConnectionStatus('error');
    };
}

function attemptReconnect() {
    if (reconnectAttempts < maxReconnectAttempts && stream) {
        reconnectAttempts++;
        updateStatus(`🔄 Reintentando conexión (${reconnectAttempts}/${maxReconnectAttempts})...`, 'warning');
        updateConnectionStatus('reconnecting');
        
        setTimeout(() => {
            if (stream) { // Only reconnect if camera is still active
                connectWebSocket();
            }
        }, Math.min(2000 * reconnectAttempts, 10000)); // Max 10 seconds
    } else {
        updateStatus('❌ No se pudo conectar al servidor. Verifica tu conexión e intenta recargar la página.', 'error');
        updateConnectionStatus('error');
    }
}

function handleWebSocketMessage(data) {
    // Log completo de datos recibidos para debug
    console.log('📨 DATOS RECIBIDOS DEL SERVIDOR:', data);
    console.log(`🔍 TIMING DEBUG: isProcessingVideoUpload=${isProcessingVideoUpload}, data.source=${data.source}, data.status="${data.status}"`);
    
    // Handle video upload responses
    if (data.source === 'upload') {
        console.log('🔍 FRONTEND DEBUG: Mensaje identificado como upload, pasando a handleVideoUploadResponse');
        handleVideoUploadResponse(data);
        return;
    }
    
    // Handle camera restoration after video upload
    if (data.type === 'camera_restored') {
        console.log('📷 FRONTEND: Cámara restaurada después de video upload');
        // No mostrar guidance messages al restaurar la cámara
        return;
    }
    
    // Handle camera blocked during video upload
    if (data.source === 'camera_blocked') {
        console.log('📷 FRONTEND: Cámara bloqueada durante video upload - ignorando mensaje');
        // No procesar mensajes de cámara bloqueada
        return;
    }
    
    // Handle camera messages during video upload - ignore them completely
    if (data.source === 'camera' && isProcessingVideoUpload) {
        console.log('📷 FRONTEND: IGNORANDO mensaje de cámara durante video upload:', data.status);
        return;
    }
    
    // Check for insufficient frames error or warnings (21 frames mínimo) - SOLO para mensajes de cámara
    if (data.source === 'camera' && data.error && (data.error.includes('insufficient frames') || data.error.includes('frames insuficientes') || 
        (data.frame_count !== undefined && data.frame_count < 21))) {
        console.log('❌ FRONTEND: Error de frames insuficientes para cámara');
        handleInsufficientFramesError(data);
        return;
    }
    
    // Check for frame count in predictions and validate (21 frames mínimo) - SOLO para cámara
    if (data.source === 'camera' && data.predictions && data.frames_count !== undefined && data.frames_count < 21) {
        console.log('❌ FRONTEND: Predicción de cámara con frames insuficientes');
        const errorData = {
            frame_count: data.frames_count,
            error: 'frames insuficientes'
        };
        handleInsufficientFramesError(errorData);
        return;
    }
    
    // Intercept low frame count from status messages - SOLO para cámara
    if (data.source === 'camera' && data.status && data.status.includes('frames capturados')) {
        const frameMatch = data.status.match(/(\d+)\s+frames\s+capturados/);
        if (frameMatch) {
            const frameCount = parseInt(frameMatch[1]);
            if (frameCount < 21) {
                console.log('❌ FRONTEND: Status de cámara con frames insuficientes');
                const errorData = {
                    frame_count: frameCount,
                    error: 'frames insuficientes'
                };
                handleInsufficientFramesError(errorData);
                return;
            }
        }
    }
    
    // Handle countdown information SOLO cuando sea válido
    if (data.countdown_active === true && data.countdown_remaining !== undefined && data.countdown_remaining > 0) {
        console.log(`🔥 FRONTEND: Mostrando countdown visual: ${data.countdown_remaining}`);
        console.log(`🔍 DEBUG: data.countdown_active = ${data.countdown_active}`);
        console.log(`🔍 DEBUG: data.countdown_remaining = ${data.countdown_remaining}`);
        handleCountdownUpdate(data.countdown_remaining);
    } else if (data.countdown_active === false || data.countdown_remaining === 0) {
        console.log(`✅ FRONTEND: Ocultando countdown`);
        console.log(`🔍 DEBUG: data.countdown_active = ${data.countdown_active}, data.countdown_remaining = ${data.countdown_remaining}`);
        hideCountdown();
    }
    
    // Handle server status messages for visual guidance (only for camera, not video upload)
    if (data.status && !isProcessingVideoUpload) {
        console.log(`🔍 FRONTEND DEBUG: Procesando status message: "${data.status}" (isProcessingVideoUpload=${isProcessingVideoUpload})`);
        handleServerStatus(data.status, data);
    } else if (data.status && isProcessingVideoUpload) {
        console.log(`🔍 FRONTEND DEBUG: IGNORANDO status message durante video upload: "${data.status}" (isProcessingVideoUpload=${isProcessingVideoUpload})`);
    }
    
    if (data.predictions) {
        updatePredictions(data);
        predictionCount++;
        updateStats(data);
    }
    
    if (data.status && (data.status.includes('demostración') || data.status.includes('demo'))) {
        showDemoWarning();
    }
    
    if (data.error) {
        console.error('Server error:', data.error);
        updateStatus('⚠️ Error del servidor: ' + data.error, 'error');
    }
}

function handleInsufficientFramesError(data) {
    const frameCount = data.frame_count || 0;
    const minFrames = 21;  // Mínimo 21 frames requeridos
    
    // Stop camera if it's streaming
    if (isStreaming && !isProcessingVideoUpload) {
        setTimeout(() => {
            stopCamera();
        }, 1000);
    }
    
    // Show error message
    const errorTitle = 'Secuencia Insuficiente';
    const errorMessage = `
        <div style="text-align: center; color: var(--danger-color);">
            <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem; color: #f59e0b;"></i>
            <h3 style="margin-bottom: 1rem;">Frames Insuficientes para Procesar</h3>
            <p style="margin-bottom: 1rem;">
                Se detectaron <strong>${frameCount}</strong> frames, pero se necesitan al menos <strong>${minFrames}</strong> frames para realizar una predicción precisa en 2.5 segundos.
            </p>
            <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4 style="color: var(--primary-color); margin-bottom: 0.5rem;">
                    <i class="fas fa-lightbulb"></i> Consejos:
                </h4>
                <ul style="text-align: left; padding-left: 1.5rem;">
                    <li>Mantén las manos visibles durante toda la seña</li>
                    <li>Realiza movimientos más lentos y claros</li>
                    <li>Asegúrate de tener buena iluminación</li>
                    <li>Evita movimientos bruscos o muy rápidos</li>
                </ul>
            </div>
            <p style="color: var(--text-secondary);">
                <strong>Intenta nuevamente</strong> realizando la seña de forma más pausada.
            </p>
        </div>
    `;
    
    showCustomModal(errorTitle, errorMessage);
    updateStatus(`❌ Frames insuficientes: ${frameCount}/${minFrames} - Intenta nuevamente`, 'error');
    
    // Clear any existing predictions display - SOLO si no hay video upload activo
    if (!isProcessingVideoUpload) {
        hideCurrentPredictionDisplay();
        console.log('🙈 Ocultando predicción por frames insuficientes');
    } else {
        console.log('🛡️ NO ocultando predicción durante video upload');
    }
}

function handleServerStatus(status, data) {
    // Handle different server states for visual feedback
    if (status.includes('Pausa después de predicción') || status.includes('⏸️')) {
        hideGuidanceMessage();
        hideCountdown();
        hideRecordingIcon();
        showPauseOverlay(status);
    } else if (status.includes('Muestra ambas manos frente a la cámara') || status.includes('👋')) {
        showGuidanceMessage('Muestra ambas manos frente a la cámara');
        hideCountdown();
        hideRecordingIcon();
        hidePauseOverlay();
    } else if (status.includes('Ambas manos detectadas') || status.includes('✋')) {
        hideGuidanceMessage();
        hideCountdown();
        hideRecordingIcon();
        hidePauseOverlay();
    } else if (status.includes('Preparándose para grabar') || status.includes('🔥')) {
        hideGuidanceMessage();
        startCountdownDisplay();
        hideRecordingIcon();
        hidePauseOverlay();
    } else if (status.includes('Grabando') || status.includes('🎬')) {
        hideGuidanceMessage();
        hideCountdown();
        showRecordingIcon();
        hidePauseOverlay();
        
        // Update recording progress if available
        if (data.recording_progress !== undefined) {
            updateRecordingProgress(data.recording_progress);
        }
    } else if (status.includes('Predicción completada') || status.includes('✅')) {
        hideGuidanceMessage();
        hideCountdown();
        hideRecordingIcon();
        hidePauseOverlay();
        
        // Auto-stop camera immediately after prediction completion
        stopCamera();
    }
    
    // Update the main status message
    updateStatus(status, getStatusType(status));
}

function getStatusType(status) {
    if (status.includes('✅')) return 'success';
    if (status.includes('⚠️') || status.includes('🔥')) return 'warning';
    if (status.includes('❌')) return 'error';
    return 'info';
}

// ===== VIDEO STREAMING =====
function startStreaming() {
    if (isStreaming && ws && ws.readyState === WebSocket.OPEN) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(function(blob) {
            if (!blob) return;
            
            const reader = new FileReader();
            reader.onload = function() {
                const base64data = reader.result.split(',')[1];
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'frame',
                        data: base64data,
                        settings: currentSettings
                    }));
                }
            };
            reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.85);
    }
    
    if (isStreaming) {
        setTimeout(startStreaming, currentSettings.frameRate);
    }
}

// ===== UI UPDATES =====
function updatePredictions(data) {
    console.log('🔍 updatePredictions llamado con:', data);
    const container = document.getElementById('predictionsContainer');
    
    if (data.predictions && data.predictions.length > 0) {
        console.log('✅ Predicciones encontradas:', data.predictions);
        console.log('🎯 Predicción principal:', data.predictions[0]);
        
        // Update the new prediction display below video
        console.log('📺 Llamando updateCurrentPredictionDisplay...');
        updateCurrentPredictionDisplay(data.predictions[0]);
        
        // Update old prediction container if it exists
        if (container) {
            // Remove placeholder
            const placeholder = container.querySelector('.predictions-placeholder');
            if (placeholder) {
                placeholder.remove();
            }
            
            // Clear existing predictions
            container.innerHTML = '';
            
            data.predictions.forEach((pred, index) => {
                const item = createPredictionItem(pred, index + 1);
                container.appendChild(item);
            });
        }
        
        // Add to history
        if (data.main_prediction && data.confidence > currentSettings.confidenceThreshold) {
            addToHistory(data.main_prediction, data.confidence);
        }
        
    } else if (data.status && data.status.includes('Recolectando')) {
        // Keep current prediction visible even when collecting
        // hideCurrentPredictionDisplay();
        if (container) {
            showLoadingInPredictions();
        }
    } else {
        // Never hide the prediction display - keep it always visible
        // Don't hide immediately - let the pause handle it
        // if (!data.status || !data.status.includes('⏸️')) {
        //     hideCurrentPredictionDisplay();
        // }
    }
}

function updateCurrentPredictionDisplay(topPrediction) {
    console.log('📺 updateCurrentPredictionDisplay llamado con:', topPrediction);
    
    const currentPrediction = document.getElementById('currentPrediction');
    const predictionText = document.getElementById('predictionText');
    const predictionConfidenceFill = document.getElementById('predictionConfidenceFill');
    const predictionConfidenceText = document.getElementById('predictionConfidenceText');
    
    console.log('🔍 Elementos DOM encontrados:', {
        currentPrediction: !!currentPrediction,
        predictionText: !!predictionText, 
        predictionConfidenceFill: !!predictionConfidenceFill,
        predictionConfidenceText: !!predictionConfidenceText
    });
    
    if (!currentPrediction || !predictionText || !predictionConfidenceFill || !predictionConfidenceText) {
        console.error('❌ Elementos DOM faltantes para mostrar predicción');
        return;
    }
    
    const confidence = topPrediction.confidence;
    const confidencePercent = (confidence * 100).toFixed(1);
    
    // Force animation reset by removing transition first
    currentPrediction.style.transition = 'none';
    currentPrediction.style.opacity = '0';
    currentPrediction.style.transform = 'translateY(20px)';
    
    // Update content
    predictionText.textContent = topPrediction.label;
    predictionConfidenceFill.style.width = `${confidencePercent}%`;
    predictionConfidenceText.textContent = `${confidencePercent}%`;
    
    // Show the prediction display
    console.log('✅ Mostrando predicción en el DOM');
    currentPrediction.style.display = 'block';
    
    // Trigger reflow to ensure the opacity/transform reset is applied
    currentPrediction.offsetHeight;
    
    // Re-enable transition and animate in
    setTimeout(() => {
        currentPrediction.style.transition = 'all 0.3s ease';
        currentPrediction.style.opacity = '1';
        currentPrediction.style.transform = 'translateY(0)';
        console.log('🎬 Animación de predicción completada');
    }, 50);
}

function hideCurrentPredictionDisplay() {
    const currentPrediction = document.getElementById('currentPrediction');
    if (currentPrediction) {
        currentPrediction.style.display = 'none';
    }
}

function createPredictionItem(prediction, rank) {
    const item = document.createElement('div');
    item.className = 'prediction-item';
    item.dataset.confidence = prediction.confidence;
    item.dataset.timestamp = Date.now();
    
    const confidencePercent = (prediction.confidence * 100).toFixed(1);
    const confidenceClass = getConfidenceClass(prediction.confidence);
    
    item.innerHTML = `
        <div class="prediction-header">
            <span class="prediction-label">#${rank} ${prediction.label}</span>
            <span class="prediction-confidence ${confidenceClass}">${confidencePercent}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
        </div>
    `;
    
    // Add animation
    item.style.opacity = '0';
    item.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        item.style.transition = 'all 0.3s ease';
        item.style.opacity = '1';
        item.style.transform = 'translateY(0)';
    }, 50);
    
    return item;
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'high-confidence';
    if (confidence >= 0.6) return 'medium-confidence';
    return 'low-confidence';
}

function showLoadingInPredictions() {
    const container = document.getElementById('predictionsContainer');
    container.innerHTML = `
        <div class="predictions-placeholder">
            <div class="placeholder-content">
                <div class="spinner"></div>
                <h3>Procesando...</h3>
                <p>Recolectando datos de las manos...</p>
            </div>
        </div>
    `;
}

function clearPredictionsDisplay() {
    const container = document.getElementById('predictionsContainer');
    container.innerHTML = `
        <div class="predictions-placeholder">
            <div class="placeholder-content">
                <i class="fas fa-hand-paper placeholder-icon"></i>
                <h3>Sin Predicciones</h3>
                <p>Muestra tus manos frente a la cámara para comenzar a detectar señas</p>
            </div>
        </div>
    `;
    
    // Hide subtitle when clearing predictions
    hidePredictionSubtitle();
}

function updateStatus(message, type) {
    // Status section was removed from UI
    // Keep function for compatibility but don't display anywhere
    console.log(`[${type.toUpperCase()}] ${message}`);
}

function getStatusIcon(type) {
    switch(type) {
        case 'success': return 'fa-check-circle';
        case 'warning': return 'fa-exclamation-triangle';
        case 'error': return 'fa-times-circle';
        default: return 'fa-info-circle';
    }
}

function updateConnectionStatus(status) {
    const indicator = document.getElementById('connectionStatus');
    if (!indicator) return;
    
    const icon = indicator.querySelector('i');
    const text = indicator.querySelector('span');
    
    indicator.className = `status-indicator ${status}`;
    
    switch(status) {
        case 'connected':
            text.textContent = 'Conectado';
            icon.className = 'fas fa-circle';
            break;
        case 'connecting':
        case 'reconnecting':
            text.textContent = 'Conectando...';
            icon.className = 'fas fa-circle';
            break;
        case 'disconnected':
            text.textContent = 'Desconectado';
            icon.className = 'fas fa-circle';
            break;
        case 'error':
            text.textContent = 'Error';
            icon.className = 'fas fa-circle';
            break;
    }
}

function updateStats(data) {
    // Update buffer size
    const bufferSize = document.getElementById('bufferSize');
    if (bufferSize && data.buffer_size !== undefined) {
        bufferSize.textContent = data.buffer_size;
    }
    
    // Update predictions count
    const predictionsCountEl = document.getElementById('predictionsCount');
    if (predictionsCountEl) {
        predictionsCountEl.textContent = predictionCount;
    }
}

function updateSessionTimer() {
    if (!sessionStartTime) return;
    
    const elapsed = Date.now() - sessionStartTime;
    const minutes = Math.floor(elapsed / 60000);
    const seconds = Math.floor((elapsed % 60000) / 1000);
    
    const timerEl = document.getElementById('sessionTime');
    if (timerEl) {
        timerEl.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
}

function updateButtonStates(cameraActive) {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    if (startBtn) startBtn.disabled = cameraActive;
    if (stopBtn) stopBtn.disabled = !cameraActive;
}

// ===== SETTINGS MANAGEMENT =====
function loadSettings() {
    const saved = localStorage.getItem('aiSignLangSettings');
    if (saved) {
        currentSettings = { ...currentSettings, ...JSON.parse(saved) };
    }
}

function saveSettings() {
    localStorage.setItem('aiSignLangSettings', JSON.stringify(currentSettings));
}

function updateSettingsUI() {
    const confidenceSlider = document.getElementById('confidenceThreshold');
    const confidenceValue = document.getElementById('confidenceValue');
    const predictionSelect = document.getElementById('predictionCount');
    const frameRateSelect = document.getElementById('frameRate');
    
    if (confidenceSlider) confidenceSlider.value = currentSettings.confidenceThreshold;
    if (confidenceValue) confidenceValue.textContent = currentSettings.confidenceThreshold;
    if (predictionSelect) predictionSelect.value = currentSettings.predictionCount;
    if (frameRateSelect) frameRateSelect.value = currentSettings.frameRate;
}

function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    if (panel) {
        panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }
}

// ===== PREDICTION FILTERING =====
function filterPredictions(filter) {
    const items = document.querySelectorAll('.prediction-item');
    
    items.forEach(item => {
        const confidence = parseFloat(item.dataset.confidence);
        
        let show = true;
        
        switch(filter) {
            case 'high':
                show = confidence >= 0.8;
                break;
            case 'all':
            default:
                show = true;
        }
        
        item.style.display = show ? 'block' : 'none';
    });
}

// ===== HISTORY MANAGEMENT =====
function addToHistory(label, confidence) {
    const historyItem = {
        label,
        confidence,
        timestamp: new Date().toLocaleTimeString()
    };
    
    // Remove placeholder if it exists
    const historyList = document.getElementById('historyList');
    const placeholder = historyList.querySelector('.history-placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    predictionHistory.unshift(historyItem);
    
    // Keep only last 50 items
    if (predictionHistory.length > 50) {
        predictionHistory = predictionHistory.slice(0, 50);
    }
    
    updateHistoryDisplay();
    
    // Show history section if hidden
    const historySection = document.getElementById('historySection');
    if (historySection && predictionHistory.length === 1) {
        historySection.style.display = 'block';
    }
}

function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    historyList.innerHTML = '';
    
    predictionHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <span class="history-label">${item.label}</span>
            <span class="history-time">${item.timestamp}</span>
        `;
        historyList.appendChild(historyItem);
    });
}

function clearHistory() {
    predictionHistory = [];
    
    // Show placeholder when history is empty
    const historyList = document.getElementById('historyList');
    if (historyList) {
        historyList.innerHTML = `
            <div class="history-placeholder">
                <div class="placeholder-content">
                    <i class="fas fa-clock placeholder-icon"></i>
                    <h3>Sin Historial</h3>
                    <p>Las predicciones aparecerán aquí conforme las realices</p>
                </div>
            </div>
        `;
    }
}

// ===== PREDICTION SUBTITLE FUNCTIONS =====
function updatePredictionSubtitle(topPrediction) {
    if (!topPrediction) {
        hidePredictionSubtitle();
        return;
    }
    
    const subtitle = document.getElementById('predictionSubtitle');
    const subtitleText = document.getElementById('subtitleText');
    const subtitleConfidence = document.getElementById('subtitleConfidence');
    const subtitleConfidenceFill = document.getElementById('subtitleConfidenceFill');
    
    if (!subtitle || !subtitleText || !subtitleConfidence || !subtitleConfidenceFill) {
        return;
    }
    
    const confidence = topPrediction.confidence;
    const confidencePercent = (confidence * 100).toFixed(1);
    
    // Update text and confidence
    subtitleText.textContent = topPrediction.label;
    subtitleConfidence.textContent = `${confidencePercent}%`;
    subtitleConfidenceFill.style.width = `${confidencePercent}%`;
    
    // Update styling based on confidence level
    subtitle.className = 'prediction-subtitle';
    if (confidence >= 0.8) {
        subtitle.classList.add('high-confidence', 'active');
    } else if (confidence >= 0.6) {
        subtitle.classList.add('medium-confidence', 'active');
    } else {
        subtitle.classList.add('low-confidence');
    }
    
    // Show subtitle with animation
    if (subtitle.style.display === 'none') {
        subtitle.style.display = 'block';
        // Trigger reflow to ensure animation plays
        subtitle.offsetHeight;
    }
    
    // Add pulse animation for high confidence predictions
    if (confidence >= currentSettings.confidenceThreshold) {
        subtitle.classList.add('active');
    } else {
        subtitle.classList.remove('active');
    }
}

function hidePredictionSubtitle() {
    const subtitle = document.getElementById('predictionSubtitle');
    if (subtitle) {
        subtitle.style.display = 'none';
        subtitle.classList.remove('high-confidence', 'medium-confidence', 'low-confidence', 'active');
    }
}

function showPredictionSubtitle() {
    const subtitle = document.getElementById('predictionSubtitle');
    if (subtitle) {
        subtitle.style.display = 'block';
    }
}

// ===== UTILITY FUNCTIONS =====
function clearPredictions() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'clear' }));
        updateStatus('🧹 Buffer limpiado', 'success');
        clearPredictionsDisplay();
    }
}

function showVideoOverlay() {
    const overlay = document.getElementById('videoOverlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
}

function hideVideoOverlay() {
    const overlay = document.getElementById('videoOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// ===== GUIDANCE MESSAGE FUNCTIONS =====
function showGuidanceMessage(message = 'Muestra ambas manos frente a la cámara') {
    console.log(`🚨 GUIDANCE MESSAGE LLAMADO: "${message}" - isProcessingVideoUpload=${isProcessingVideoUpload}`);
    console.trace('📍 STACK TRACE para showGuidanceMessage:');
    
    // PROTECCIÓN ABSOLUTA: No mostrar guidance durante video upload
    if (isProcessingVideoUpload) {
        console.log('🚫 GUIDANCE BLOQUEADO: Video upload en progreso - no mostrando guidance message');
        return;
    }
    
    const overlay = document.getElementById('guidanceMessageOverlay');
    const textElement = document.getElementById('guidanceText');
    
    if (overlay && textElement) {
        textElement.textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideGuidanceMessage() {
    console.log('🙈 OCULTANDO guidance message');
    const overlay = document.getElementById('guidanceMessageOverlay');
    if (overlay) {
        overlay.style.display = 'none';
        console.log('✅ Guidance message ocultado exitosamente');
    } else {
        console.log('❌ No se encontró el overlay para ocultar');
    }
}

function forceHideGuidanceMessage() {
    // Fuerza ocultar guidance message sin importar el estado
    const overlay = document.getElementById('guidanceMessageOverlay');
    if (overlay && overlay.style.display !== 'none') {
        console.log('🚫 FORZANDO ocultamiento de guidance message durante video upload');
        overlay.style.display = 'none';
    }
}

function startGuidanceBlocking() {
    console.log('🛡️ INICIANDO bloqueo de guidance message durante video upload');
    // Forzar ocultar cada 100ms durante video upload
    guidanceBlockInterval = setInterval(forceHideGuidanceMessage, 100);
}

function stopGuidanceBlocking() {
    console.log('🛡️ TERMINANDO bloqueo de guidance message');
    if (guidanceBlockInterval) {
        clearInterval(guidanceBlockInterval);
        guidanceBlockInterval = null;
    }
}

function cleanupVideoUploadConnection() {
    console.log('🧹 VIDEO UPLOAD: Limpiando conexión después de video upload');
    
    // Si no hay cámara activa, cerrar la conexión WebSocket
    if (!isStreaming && ws && ws.readyState === WebSocket.OPEN) {
        console.log('📡 VIDEO UPLOAD: Cerrando conexión WebSocket (no hay cámara activa)');
        ws.close();
        ws = null;
    } else if (isStreaming) {
        console.log('📡 VIDEO UPLOAD: Manteniendo conexión WebSocket (cámara activa)');
    }
}

// ===== COUNTDOWN FUNCTIONS =====
let countdownTimer = null;
let currentCountdownNumber = 3;

function startCountdownDisplay() {
    const overlay = document.getElementById('countdownOverlay');
    const numberElement = document.getElementById('countdownNumber');
    
    if (!overlay || !numberElement) return;
    
    // Clear any existing timer
    if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
    }
    
    // Reset countdown
    currentCountdownNumber = 3;
    
    // Show overlay and display initial number immediately
    overlay.style.display = 'flex';
    numberElement.textContent = currentCountdownNumber;
    
    // Trigger initial animation
    numberElement.style.animation = 'none';
    numberElement.offsetHeight; // Force reflow
    numberElement.style.animation = 'countdownPulse 1s ease-in-out';
    
    // Start countdown timer
    countdownTimer = setInterval(() => {
        currentCountdownNumber--;
        
        if (currentCountdownNumber > 0) {
            // Show next number
            numberElement.textContent = currentCountdownNumber;
            
            // Trigger animation
            numberElement.style.animation = 'none';
            numberElement.offsetHeight; // Force reflow
            numberElement.style.animation = 'countdownPulse 1s ease-in-out';
        } else {
            // Countdown finished
            clearInterval(countdownTimer);
            countdownTimer = null;
            setTimeout(() => {
                hideCountdown();
            }, 300); // Short delay before hiding
        }
    }, 1000);
}

function hideCountdown() {
    console.log('✅ OCULTANDO countdown sobre cámara');
    
    // Ocultar el nuevo countdown sobre la cámara
    const cameraOverlay = document.getElementById('countdownCameraOverlay');
    if (cameraOverlay) {
        cameraOverlay.style.display = 'none';
    }
    
    // Legacy: Ocultar el countdown de pantalla completa (por compatibilidad)
    const overlay = document.getElementById('countdownOverlay');
    if (overlay) {
        overlay.style.display = 'none';
        overlay.style.visibility = 'hidden';
        // Limpiar estilos forzados
        overlay.style.position = '';
        overlay.style.top = '';
        overlay.style.left = '';
        overlay.style.width = '';
        overlay.style.height = '';
        overlay.style.zIndex = '';
    }
    
    // Clear timer if running
    if (countdownTimer) {
        clearInterval(countdownTimer);
        countdownTimer = null;
    }
}

function handleCountdownUpdate(remainingSeconds) {
    console.log(`🔍 DEBUG: handleCountdownUpdate llamado con ${remainingSeconds}`);
    
    const overlay = document.getElementById('countdownCameraOverlay');
    const numberElement = document.getElementById('countdownCameraNumber');
    const messageElement = document.getElementById('countdownCameraMessage');
    
    console.log(`🔍 DEBUG: Elementos encontrados - overlay: ${!!overlay}, number: ${!!numberElement}, message: ${!!messageElement}`);
    
    if (!overlay || !numberElement || !messageElement) {
        console.error('🔴 ERROR: Elementos del countdown no encontrados');
        console.error('overlay:', overlay);
        console.error('numberElement:', numberElement);
        console.error('messageElement:', messageElement);
        return;
    }
    
    if (remainingSeconds > 0) {
        console.log(`🔥 COUNTDOWN VISUAL SOBRE CÁMARA: ${remainingSeconds}`);
        
        // Mostrar overlay sobre la cámara con estilos forzados
        overlay.style.display = 'flex';
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.right = '0';
        overlay.style.bottom = '0';
        overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.75)';
        overlay.style.zIndex = '9999';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';
        
        console.log(`🔍 DEBUG: Overlay display set to: ${overlay.style.display}`);
        console.log(`🔍 DEBUG: Overlay computed style:`, window.getComputedStyle(overlay).display);
        
        // Actualizar número con estilos forzados
        numberElement.textContent = remainingSeconds;
        numberElement.style.fontSize = '8rem';
        numberElement.style.color = '#6366f1';
        numberElement.style.fontWeight = '900';
        numberElement.style.textAlign = 'center';
        numberElement.style.display = 'block';
        
        // Actualizar mensaje dinámico
        messageElement.textContent = `Iniciando grabación en ${remainingSeconds} segundo${remainingSeconds !== 1 ? 's' : ''}...`;
        messageElement.style.fontSize = '1.8rem';
        messageElement.style.color = 'white';
        messageElement.style.textAlign = 'center';
        
        // Trigger animation
        numberElement.style.animation = 'none';
        numberElement.offsetHeight; // Force reflow
        numberElement.style.animation = 'countdownPulse 1s ease-in-out';
        
        console.log(`✅ DEBUG: Countdown actualizado visualmente - ${remainingSeconds}`);
    } else {
        console.log(`🔥 DEBUG: remainingSeconds <= 0, ocultando countdown`);
        // Hide countdown when finished
        hideCountdown();
    }
}

// ===== FUNCIÓN DE TEST PARA COUNTDOWN =====
function testCountdownDisplay() {
    console.log('🧪 TEST: Probando countdown visual...');
    
    const overlay = document.getElementById('countdownCameraOverlay');
    console.log('🧪 TEST: Overlay encontrado:', !!overlay);
    
    if (overlay) {
        overlay.style.display = 'flex';
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.right = '0';
        overlay.style.bottom = '0';
        overlay.style.backgroundColor = 'rgba(255, 0, 0, 0.8)'; // Rojo para debug
        overlay.style.zIndex = '9999';
        overlay.style.alignItems = 'center';
        overlay.style.justifyContent = 'center';
        
        const numberElement = document.getElementById('countdownCameraNumber');
        if (numberElement) {
            numberElement.textContent = 'TEST';
            numberElement.style.fontSize = '8rem';
            numberElement.style.color = 'white';
            numberElement.style.fontWeight = '900';
        }
        
        const messageElement = document.getElementById('countdownCameraMessage');
        if (messageElement) {
            messageElement.textContent = 'TEST MESSAGE';
            messageElement.style.fontSize = '2rem';
            messageElement.style.color = 'white';
        }
        
        console.log('🧪 TEST: Countdown mostrado con fondo rojo');
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            overlay.style.display = 'none';
            console.log('🧪 TEST: Countdown ocultado');
        }, 3000);
    }
}

// Agregar función de test al window para poder llamarla desde la consola
window.testCountdownDisplay = testCountdownDisplay;

// ===== RECORDING ICON FUNCTIONS =====
function showRecordingIcon() {
    const icon = document.getElementById('recordingIcon');
    if (icon) {
        icon.style.display = 'flex';
    }
}

function hideRecordingIcon() {
    const icon = document.getElementById('recordingIcon');
    if (icon) {
        icon.style.display = 'none';
    }
}

function updateRecordingProgress(progress) {
    // Optional: Could show progress in the recording icon
    // For now, just ensure the icon is visible during recording
    showRecordingIcon();
}

// ===== PAUSE OVERLAY FUNCTIONS =====
function showPauseOverlay(statusMessage) {
    const overlay = document.getElementById('pauseOverlay');
    const timerElement = document.getElementById('pauseTimer');
    
    if (!overlay || !timerElement) return;
    
    // Extract remaining time from status message
    const timeMatch = statusMessage.match(/(\d+\.\d+)s/);
    if (timeMatch) {
        timerElement.textContent = timeMatch[1] + 's';
    }
    
    overlay.style.display = 'flex';
}

function hidePauseOverlay() {
    const overlay = document.getElementById('pauseOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showDemoWarning() {
    const warning = document.getElementById('demoWarning');
    if (warning) {
        warning.style.display = 'flex';
    }
}

function showLoadingOverlay(message = 'Cargando...') {
    const overlay = document.getElementById('loadingOverlay');
    const messageEl = document.getElementById('loadingMessage');
    
    if (overlay) overlay.style.display = 'flex';
    if (messageEl) messageEl.textContent = message;
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function toggleFullscreen() {
    const videoContainer = document.querySelector('.video-container');
    
    if (!document.fullscreenElement) {
        videoContainer.requestFullscreen().catch(err => {
            console.error('Error entering fullscreen:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

// ===== KEYBOARD SHORTCUTS =====
function handleKeyboardShortcuts(event) {
    // Prevent shortcuts when typing in inputs
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') {
        return;
    }
    
    switch(event.key) {
        case ' ':
            event.preventDefault();
            if (stream) {
                stopCamera();
            } else {
                startCamera();
            }
            break;
        case 'c':
        case 'C':
            clearPredictions();
            break;
        case 's':
        case 'S':
            toggleSettings();
            break;
        case 'f':
        case 'F':
            toggleFullscreen();
            break;
        case 'Escape':
            if (document.fullscreenElement) {
                document.exitFullscreen();
            }
            break;
    }
}

// ===== WINDOW EVENT HANDLERS =====
function handleWindowFocus() {
    // Resume streaming if it was interrupted
    if (stream && ws && ws.readyState === WebSocket.OPEN && !isStreaming) {
        isStreaming = true;
        startStreaming();
    }
}

function handleWindowBlur() {
    // Optionally pause streaming when window loses focus to save resources
    // isStreaming = false;
}

// ===== MODAL FUNCTIONS =====
function showAbout() {
    const modal = document.getElementById('aboutModal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

function showHelp() {
    const helpContent = `
        <h3><i class="fas fa-keyboard"></i> Atajos de Teclado</h3>
        <ul>
            <li><strong>Espacio:</strong> Iniciar/Detener cámara</li>
            <li><strong>C:</strong> Limpiar buffer</li>
            <li><strong>S:</strong> Mostrar/Ocultar configuración</li>
            <li><strong>F:</strong> Pantalla completa</li>
            <li><strong>Escape:</strong> Salir de pantalla completa</li>
        </ul>
        
        <h3><i class="fas fa-lightbulb"></i> Consejos de Uso</h3>
        <ul>
            <li>Asegúrate de tener buena iluminación</li>
            <li>Mantén las manos visibles en el encuadre</li>
            <li>Evita fondos muy complejos</li>
            <li>Realiza los gestos de forma clara y pausada</li>
        </ul>
    `;
    
    showCustomModal('Ayuda', helpContent);
}

function showCustomModal(title, content) {
    // Create a temporary modal for custom content
    const modalHtml = `
        <div class="modal" id="customModal" style="display: flex;">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${title}</h3>
                    <button class="modal-close" onclick="closeModal('customModal')">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
            </div>
        </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

function showErrorModal(title, message) {
    const errorContent = `
        <div style="text-align: center; color: var(--danger-color);">
            <i class="fas fa-exclamation-triangle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
            <p>${message}</p>
        </div>
    `;
    
    showCustomModal(title, errorContent);
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
        
        // Remove custom modals from DOM
        if (modalId === 'customModal') {
            modal.remove();
        }
    }
}

// ===== VIDEO UPLOAD FUNCTIONALITY =====
function uploadVideo() {
    const fileInput = document.getElementById('videoFileInput');
    fileInput.click();
}

async function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
        isProcessingVideoUpload = true;
        
        // Ocultar explícitamente guidance message al iniciar video upload
        hideGuidanceMessage();
        
        // Iniciar bloqueo activo de guidance message
        startGuidanceBlocking();
        
        showLoadingOverlay('Procesando video...');
        updateStatus('📹 Procesando video subido...', 'info');
        
        // Reset video upload processor on server
        if (ws && ws.readyState === WebSocket.OPEN) {
            console.log('🔄 VIDEO UPLOAD: Enviando reset al servidor');
            ws.send(JSON.stringify({
                type: 'reset_video_upload',
                source: 'upload_init'  // Identificar como inicialización de upload
            }));
        }
        
        // Create video element for processing
        const videoElement = document.createElement('video');
        videoElement.src = URL.createObjectURL(file);
        videoElement.muted = true;
        
        videoElement.onloadedmetadata = async function() {
            try {
                await processVideoFrames(videoElement);
            } catch (err) {
                console.error('Error processing video:', err);
                updateStatus('❌ Error procesando video: ' + err.message, 'error');
                showErrorModal('Error de Video', 'No se pudo procesar el video. Asegúrate de que sea un formato válido.');
            } finally {
                isProcessingVideoUpload = false;
                stopGuidanceBlocking();  // Terminar bloqueo activo
                cleanupVideoUploadConnection();  // Limpiar conexión si es necesario
                hideLoadingOverlay();
                URL.revokeObjectURL(videoElement.src);
            }
        };
        
        videoElement.onerror = function() {
            isProcessingVideoUpload = false;
            stopGuidanceBlocking();  // Terminar bloqueo activo
            cleanupVideoUploadConnection();  // Limpiar conexión si es necesario
            hideLoadingOverlay();
            updateStatus('❌ Error cargando video', 'error');
            showErrorModal('Error de Video', 'No se pudo cargar el video. Verifica que sea un formato compatible.');
        };
        
    } catch (err) {
        console.error('Error uploading video:', err);
        isProcessingVideoUpload = false;
        stopGuidanceBlocking();  // Terminar bloqueo activo
        cleanupVideoUploadConnection();  // Limpiar conexión si es necesario
        hideLoadingOverlay();
        updateStatus('❌ Error subiendo video', 'error');
        showErrorModal('Error', 'Error al procesar el archivo de video.');
    }
}

async function processVideoFrames(videoElement) {
    return new Promise((resolve, reject) => {
        // Nota: No pausamos la cámara para evitar interferir con el flujo
        const wasStreaming = isStreaming;
        
        // Connect to WebSocket if not connected - MODO VIDEO UPLOAD
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.log('📡 VIDEO UPLOAD: Estableciendo conexión WebSocket independiente');
            connectWebSocketForVideoUpload();
            
            // Wait for connection
            const checkConnection = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    clearInterval(checkConnection);
                    console.log('✅ VIDEO UPLOAD: Conexión WebSocket establecida');
                    startVideoFrameExtraction(videoElement, resolve, reject, wasStreaming);
                }
            }, 100);
            
            // Timeout after 10 seconds
            setTimeout(() => {
                clearInterval(checkConnection);
                reject(new Error('No se pudo conectar al servidor para video upload'));
            }, 10000);
        } else {
            console.log('🔄 VIDEO UPLOAD: Reutilizando conexión WebSocket existente');
            startVideoFrameExtraction(videoElement, resolve, reject, wasStreaming);
        }
    });
}

function startVideoFrameExtraction(videoElement, resolve, reject, wasStreaming) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 640;  // Standard width for processing
    canvas.height = 480; // Standard height for processing
    
    const targetFrames = 50; // Exactamente 50 frames
    const videoDuration = videoElement.duration;
    const frameInterval = videoDuration / targetFrames;
    
    console.log(`📹 VIDEO UPLOAD: Extrayendo ${targetFrames} frames de video de ${videoDuration.toFixed(2)}s`);
    
    // Validación básica de duración
    if (videoDuration < 1.0) {
        reject(new Error(`Video muy corto: ${videoDuration.toFixed(2)}s - Se necesita al menos 1 segundo`));
        return;
    }
    
    let currentTime = 0;
    let framesProcessed = 0;
    
    const extractNextFrame = () => {
        if (framesProcessed >= targetFrames) {
            console.log(`✅ VIDEO UPLOAD: Extracción completada - ${framesProcessed} frames enviados`);
            updateStatus('✅ Frames extraídos - Esperando predicción...', 'info');
            
            // Enviar mensaje de finalización al backend para trigger final de predicción
            if (ws && ws.readyState === WebSocket.OPEN) {
                console.log('🏁 VIDEO UPLOAD: Enviando mensaje de finalización al servidor');
                ws.send(JSON.stringify({
                    type: 'video_upload_finished',
                    total_frames: targetFrames,
                    settings: currentSettings,
                    source: 'upload_finish'  // Identificar como finalización de upload
                }));
            }
            
            resolve();
            return;
        }
        
        videoElement.currentTime = currentTime;
    };
    
    videoElement.onseeked = function() {
        // Draw current frame to canvas
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert to blob and send via WebSocket
        canvas.toBlob(function(blob) {
            if (!blob) return;
            
            const reader = new FileReader();
            reader.onload = function() {
                const base64data = reader.result.split(',')[1];
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'frame',
                        data: base64data,
                        settings: currentSettings,
                        source: 'upload'
                    }));
                }
                
                framesProcessed++;
                currentTime += frameInterval;
                
                // Update progress locally
                const progress = (framesProcessed / targetFrames) * 100;
                updateStatus(`📹 Extrayendo frames: ${framesProcessed}/${targetFrames} (${Math.round(progress)}%)`, 'info');
                
                // Extract next frame
                setTimeout(extractNextFrame, 50); // Faster extraction
            };
            reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.85);
    };
    
    // Start extraction
    extractNextFrame();
}

function handleVideoUploadResponse(data) {
    console.log('📹 VIDEO UPLOAD RESPONSE:', data);
    
    // Update progress if available
    if (data.upload_progress) {
        const progress = data.upload_progress;
        const progressPercent = Math.round(progress.progress_percent);
        
        updateStatus(`📹 Procesando video... ${progress.frames_with_hands}/${progress.target_frames} frames (${progressPercent}%)`, 'info');
        
        // Update loading overlay message
        const loadingMessage = document.getElementById('loadingMessage');
        if (loadingMessage) {
            loadingMessage.textContent = `Extrayendo keypoints: ${progress.frames_with_hands}/${progress.target_frames} frames (${progressPercent}%)`;
        }
    }
    
    // Handle prediction results
    if (data.predictions) {
        console.log('🎯 VIDEO UPLOAD: Predicción completada!');
        console.log('🎯 VIDEO UPLOAD: Datos de predicción:', data);
        console.log('🎯 VIDEO UPLOAD: Llamando updatePredictions...');
        updatePredictions(data);
        predictionCount++;
        updateStats(data);
        
        // Add to history
        if (data.main_prediction && data.confidence > currentSettings.confidenceThreshold) {
            addToHistory(data.main_prediction, data.confidence);
        }
        
        updateStatus(`✅ Video procesado: ${data.main_prediction} (${(data.confidence * 100).toFixed(1)}%)`, 'success');
        
        // Hide loading overlay
        hideLoadingOverlay();
        
        // Reset flag after a longer delay to ensure all related messages are processed
        setTimeout(() => {
            isProcessingVideoUpload = false;
            stopGuidanceBlocking();  // Terminar bloqueo activo
            cleanupVideoUploadConnection();  // Limpiar conexión si es necesario
            console.log('📷 FRONTEND: Video upload completado - Cámara disponible nuevamente');
        }, 2000);  // Aumentado a 2 segundos para evitar race conditions
    }
    
    // Handle errors
    if (data.error) {
        console.error('❌ VIDEO UPLOAD ERROR:', data.error);
        updateStatus('❌ Error procesando video: ' + data.error, 'error');
        hideLoadingOverlay();
        
        // Reset flag after delay even for errors to prevent race conditions
        setTimeout(() => {
            isProcessingVideoUpload = false;
            stopGuidanceBlocking();  // Terminar bloqueo activo
            cleanupVideoUploadConnection();  // Limpiar conexión si es necesario
            console.log('📷 FRONTEND: Video upload error - Cámara disponible nuevamente');
        }, 2000);
    }
}

// ===== CLEANUP =====
function cleanup() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    if (ws) {
        ws.close();
    }
    
    saveSettings();
}

// ===== CLICK OUTSIDE TO CLOSE MODALS =====
document.addEventListener('click', function(event) {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        if (event.target === modal) {
            closeModal(modal.id);
        }
    });
});
