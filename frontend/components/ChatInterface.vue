<template>
  <div class="flex flex-col h-screen bg-gray-100">
    <!-- Header -->
    <header class="bg-white shadow-sm p-4">
      <div class="max-w-4xl mx-auto flex justify-between items-center">
        <h1 class="text-xl font-semibold text-gray-800">RAG Chat Interface</h1>
        <div class="space-x-2">
          <label class="cursor-pointer px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
            Upload PDF
            <input type="file" class="hidden" accept=".pdf" @change="handlePdfUpload" />
          </label>
          <button
            @click="showImageUpload = true"
            class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Upload Image
          </button>
          <button
            @click="store.clearMessages()"
            class="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
          >
            Clear Chat
          </button>
        </div>
      </div>
    </header>

    <!-- Chat Messages -->
    <div class="flex-1 overflow-y-auto p-4">
      <div class="max-w-4xl mx-auto space-y-4">
        <div v-if="store.messages.length === 0" class="text-center text-gray-500 py-8">
          No messages yet. Start a conversation!
        </div>
        
        <div v-for="message in store.messages" :key="message.id" class="flex flex-col space-y-2">
          <!-- Message -->
          <div
            :class="[
              'p-4 rounded-lg max-w-3xl',
              message.type === 'user'
                ? 'bg-blue-100 ml-auto'
                : 'bg-white'
            ]"
          >
            <div class="prose" v-html="formatMessage(message.content)"></div>
          </div>

          <!-- Relevant Images -->
          <div v-if="message.type === 'assistant' && message.relevantImages?.length" class="flex flex-wrap gap-4 mt-2">
            <div v-for="image in message.relevantImages" :key="image.filename" class="relative group">
              <img
                :src="store.getImageUrl(image.filename)"
                :alt="image.filename"
                class="w-48 h-48 object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow"
              />
              <div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-opacity rounded-lg flex items-center justify-center opacity-0 group-hover:opacity-100">
                <div class="text-white text-sm p-2 text-center">
                  {{ image.filename }}
                </div>
              </div>
            </div>
          </div>

          <!-- Chunks -->
          <div v-if="message.type === 'assistant' && message.chunks?.length" class="mt-2 space-y-2">
            <div v-for="(chunk, index) in message.chunks" :key="index" class="text-sm text-gray-600 bg-gray-50 p-3 rounded">
              <div class="font-medium mb-1">Source: {{ chunk.source }} • Page: {{ chunk.page + 1 }}</div>
              <div class="whitespace-pre-wrap">{{ chunk.chunk }}</div>
            </div>
          </div>
        </div>

        <!-- Loading Indicator -->
        <div v-if="store.loading" class="flex justify-center py-4">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      </div>
    </div>

    <!-- Input Form -->
    <div class="bg-white border-t p-4">
      <div class="max-w-4xl mx-auto">
        <form @submit.prevent="handleSubmit" class="flex gap-2 items-center">
          <input
            v-model="message"
            type="text"
            placeholder="Ask a question..."
            class="flex-1 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            :disabled="store.loading"
          />
          <VoiceQuery
            @query-response="handleVoiceResponse"
            class="mx-2"
          />
          <button
            type="submit"
            class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
            :disabled="!message.trim() || store.loading"
          >
            Send
          </button>
        </form>
      </div>
    </div>

    <!-- Image Upload Modal -->
    <div v-if="showImageUpload" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
      <div class="bg-white p-6 rounded-lg max-w-md w-full">
        <h2 class="text-xl font-semibold mb-4">Upload Image</h2>
        <form @submit.prevent="handleImageUpload" class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700">Image File</label>
            <input
              type="file"
              accept="image/*"
              @change="handleFileSelect"
              class="mt-1 block w-full"
              required
            />
          </div>
          <div>
            <label class="block text-sm font-medium text-gray-700">Image ID</label>
            <input
              v-model="imageUpload.imageId"
              type="text"
              class="mt-1 block w-full border rounded p-2"
              required
            />
          </div>
          <div class="flex justify-end space-x-2">
            <button
              type="button"
              @click="showImageUpload = false"
              class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
              :disabled="!selectedFile || store.loading"
            >
              Upload
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useChatStore } from '~/stores/chat'
import VoiceQuery from './VoiceQuery.vue'

const store = useChatStore()
const message = ref('')
const showImageUpload = ref(false)
const selectedFile = ref<File | null>(null)

const imageUpload = ref({
  imageId: ''
})

const handleSubmit = async () => {
  if (!message.value.trim() || store.loading) return
  const content = message.value
  message.value = ''
  await store.sendMessage(content)
}

const handleFileSelect = (event: Event) => {
  const input = event.target as HTMLInputElement
  if (input.files?.length) {
    selectedFile.value = input.files[0]
  }
}

const handleImageUpload = async () => {
  if (!selectedFile.value) return

  try {
    await store.uploadImage(selectedFile.value, {
      imageId: imageUpload.value.imageId
    })
    showImageUpload.value = false
    selectedFile.value = null
    imageUpload.value = {
      imageId: ''
    }
  } catch (error) {
    console.error('Failed to upload image:', error)
  }
}

const handlePdfUpload = async (event: Event) => {
  const input = event.target as HTMLInputElement
  if (!input.files?.length) return

  const file = input.files[0]
  const sourceId = prompt('Enter a unique identifier for this document:')
  if (!sourceId) return

  try {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('source_id', sourceId)

    const response = await fetch(`${useRuntimeConfig().public.apiBase}/upload_pdf/`, {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error('Failed to upload PDF')
    }

    alert('PDF uploaded successfully!')
  } catch (error) {
    console.error('Failed to upload PDF:', error)
    alert('Failed to upload PDF. Please try again.')
  } finally {
    // Reset input
    input.value = ''
  }
}

const formatMessage = (content: string) => {
  return content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
}

const handleVoiceResponse = (response: any) => {
  if (response.error) {
    // Handle error if needed
    console.error('Voice query error:', response.error)
    return
  }
  
  // Add the voice query response to the chat
  store.messages.push({
    type: 'user',
    content: response.transcript || 'Voice query',
    timestamp: new Date().toISOString()
  })
  
  store.messages.push({
    type: 'assistant',
    content: response.answer,
    timestamp: new Date().toISOString(),
    chunks: response.chunks,
    relevantImages: response.relevant_images
  })
  
  // Scroll to bottom
  store.$nextTick(() => {
    const chatContainer = document.querySelector('.overflow-y-auto')
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight
    }
  })
}
</script>

<style scoped>
.prose {
  max-width: none;
}

.input-wrapper {
  display: flex;
  align-items: center;
  gap: 1rem;
  width: 100%;
}

.voice-query-wrapper {
  flex-shrink: 0;
}

/* Update existing styles to accommodate voice query */
.input-container {
  padding: 1rem;
  background-color: #f5f5f5;
  border-top: 1px solid #e0e0e0;
}

textarea {
  flex-grow: 1;
  min-height: 40px;
  max-height: 120px;
  padding: 0.5rem;
  border: 1px solid #e0e0e0;
  border-radius: 0.5rem;
  resize: vertical;
}
</style> 