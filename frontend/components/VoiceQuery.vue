<template>
  <div class="voice-query">
    <button 
      @mousedown="startRecording" 
      @mouseup="stopRecording"
      @mouseleave="stopRecording"
      :class="{ 'recording': isRecording }"
      class="voice-button"
    >
      <i class="fas" :class="isRecording ? 'fa-stop' : 'fa-microphone'"></i>
      {{ isRecording ? 'Recording...' : 'Hold to Speak' }}
    </button>
    <div v-if="error" class="error">
      {{ error }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRuntimeConfig } from 'nuxt/app'

const isRecording = ref(false)
const mediaRecorder = ref<MediaRecorder | null>(null)
const audioChunks = ref<Blob[]>([])
const ws = ref<WebSocket | null>(null)
const transcript = ref('')
const error = ref<string | null>(null)

const emit = defineEmits(['query-response'])

const connectWebSocket = () => {
  const config = useRuntimeConfig()
  const apiBase = config.public.apiBase as string
  const wsUrl = apiBase.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws/voice'
  
  try {
    ws.value = new WebSocket(wsUrl)
    
    ws.value.onopen = () => {
      console.log('WebSocket connection established')
      error.value = null
    }
    
    ws.value.onmessage = (event) => {
      const response = JSON.parse(event.data)
      if (response.error) {
        error.value = response.error
      } else {
        transcript.value = response.transcript || ''
        emit('query-response', {
          ...response,
          transcript: response.transcript
        })
      }
    }
    
    ws.value.onerror = (event) => {
      console.error('WebSocket error:', event)
      error.value = 'WebSocket connection error'
    }
    
    ws.value.onclose = () => {
      console.log('WebSocket connection closed')
      ws.value = null
    }
  } catch (err) {
    console.error('Error creating WebSocket:', err)
    error.value = 'Failed to establish WebSocket connection'
  }
}

const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder.value = new MediaRecorder(stream)
    audioChunks.value = []
    
    mediaRecorder.value.ondataavailable = (event) => {
      audioChunks.value.push(event.data)
    }
    
    mediaRecorder.value.onstop = async () => {
      const audioBlob = new Blob(audioChunks.value, { type: 'audio/wav' })
      await sendAudioToServer(audioBlob)
    }
    
    // Connect to WebSocket if not already connected
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) {
      connectWebSocket()
    }
    
    // Start recording
    mediaRecorder.value.start()
    isRecording.value = true
    error.value = null
  } catch (err) {
    console.error('Error accessing microphone:', err)
    error.value = 'Error accessing microphone: ' + (err as Error).message
  }
}

const stopRecording = () => {
  if (mediaRecorder.value && isRecording.value) {
    mediaRecorder.value.stop()
    isRecording.value = false
    
    // Stop all audio tracks
    mediaRecorder.value.stream.getTracks().forEach(track => track.stop())
  }
}

const convertToWav = async (audioBlob: Blob): Promise<Blob> => {
  const audioContext = new AudioContext()
  const arrayBuffer = await audioBlob.arrayBuffer()
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
  
  // Create WAV file
  const numChannels = audioBuffer.numberOfChannels
  const sampleRate = audioBuffer.sampleRate
  const format = 1 // PCM
  const bitDepth = 16
  
  const bytesPerSample = bitDepth / 8
  const blockAlign = numChannels * bytesPerSample
  const byteRate = sampleRate * blockAlign
  const dataSize = audioBuffer.length * blockAlign
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)
  
  // RIFF identifier
  writeString(view, 0, 'RIFF')
  // RIFF chunk length
  view.setUint32(4, 36 + dataSize, true)
  // RIFF type
  writeString(view, 8, 'WAVE')
  // format chunk identifier
  writeString(view, 12, 'fmt ')
  // format chunk length
  view.setUint32(16, 16, true)
  // sample format (raw)
  view.setUint16(20, format, true)
  // channel count
  view.setUint16(22, numChannels, true)
  // sample rate
  view.setUint32(24, sampleRate, true)
  // byte rate (sample rate * block align)
  view.setUint32(28, byteRate, true)
  // block align (channel count * bytes per sample)
  view.setUint16(32, blockAlign, true)
  // bits per sample
  view.setUint16(34, bitDepth, true)
  // data chunk identifier
  writeString(view, 36, 'data')
  // data chunk length
  view.setUint32(40, dataSize, true)
  
  // Write the PCM samples
  const offset = 44
  const channelData: Float32Array[] = []
  for (let i = 0; i < numChannels; i++) {
    channelData.push(audioBuffer.getChannelData(i))
  }
  
  let pos = 0
  while (pos < audioBuffer.length) {
    for (let i = 0; i < numChannels; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i][pos]))
      const val = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
      view.setInt16(offset + (pos * blockAlign) + (i * bytesPerSample), val, true)
    }
    pos++
  }
  
  return new Blob([buffer], { type: 'audio/wav' })
}

const writeString = (view: DataView, offset: number, string: string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i))
  }
}

const sendAudioToServer = async (audioBlob: Blob) => {
  if (ws.value?.readyState === WebSocket.OPEN) {
    try {
      // Convert to WAV format before sending
      const wavBlob = await convertToWav(audioBlob)
      const arrayBuffer = await wavBlob.arrayBuffer()
      ws.value.send(arrayBuffer)
    } catch (err) {
      console.error('Error sending audio:', err)
      error.value = 'Error sending audio: ' + (err as Error).message
    }
  } else {
    error.value = 'WebSocket connection not available'
    // Try to reconnect
    connectWebSocket()
  }
}

onBeforeUnmount(() => {
  if (ws.value) {
    ws.value.close()
  }
  if (mediaRecorder.value && isRecording.value) {
    stopRecording()
  }
})
</script>

<style scoped>
.voice-query {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
}

.voice-button {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border-radius: 2rem;
  border: none;
  background-color: #4CAF50;
  color: white;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.voice-button:hover {
  background-color: #45a049;
}

.voice-button.recording {
  background-color: #f44336;
  animation: pulse 1.5s infinite;
}

.voice-button i {
  font-size: 1.2rem;
}

.error {
  color: #f44336;
  margin-top: 0.5rem;
}

@keyframes pulse {
  0% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
  }
}

.transcript {
  display: none;
}
</style> 