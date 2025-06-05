<template>
  <div class="flex flex-col h-screen bg-gray-50">
    <!-- Header -->
    <header class="bg-white shadow-sm p-4">
      <div class="max-w-4xl mx-auto flex justify-between items-center">
        <h1 class="text-xl font-semibold text-gray-800">RAG Chat Interface</h1>
        <div class="flex gap-4">
          <label class="cursor-pointer bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors">
            Upload PDF
            <input type="file" class="hidden" accept=".pdf" @change="handleFileUpload" />
          </label>
          <button
            @click="store.clearMessages()"
            class="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors"
          >
            Clear Chat
          </button>
        </div>
      </div>
    </header>

    <!-- Chat Messages -->
    <div class="flex-1 overflow-y-auto p-4">
      <div class="max-w-4xl mx-auto space-y-4">
        <div v-if="store.messages.length === 0" class="text-center text-gray-500 mt-8">
          Start a conversation by asking a question about your documents
        </div>
        
        <template v-for="message in store.messages" :key="message.id">
          <div
            :class="[
              'flex gap-4 p-4 rounded-lg',
              message.type === 'user' ? 'bg-blue-50' : 'bg-white shadow-sm'
            ]"
          >
            <div class="w-8 h-8 rounded-full flex items-center justify-center"
              :class="message.type === 'user' ? 'bg-blue-500' : 'bg-green-500'">
              <span class="text-white font-medium">
                {{ message.type === 'user' ? 'U' : 'A' }}
              </span>
            </div>
            <div class="flex-1">
              <div class="text-sm text-gray-500 mb-1">
                {{ message.type === 'user' ? 'You' : 'Assistant' }} • 
                {{ new Date(message.timestamp).toLocaleTimeString() }}
              </div>
              <div class="prose max-w-none" v-html="formatMessage(message.content)"></div>
              
              <!-- Show chunks for assistant messages -->
              <div v-if="message.type === 'assistant' && message.chunks" class="mt-4 space-y-2">
                <div v-for="(chunk, index) in message.chunks" :key="index" 
                     class="text-sm bg-gray-50 p-3 rounded border border-gray-200">
                  <div class="text-xs text-gray-500 mb-1">
                    Source: {{ chunk.source }} • Page: {{ chunk.page + 1 }}
                  </div>
                  <div class="text-gray-700">{{ chunk.chunk }}</div>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Loading indicator -->
        <div v-if="store.isLoading" class="flex justify-center py-4">
          <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>

        <!-- Error message -->
        <div v-if="store.error" class="bg-red-50 text-red-700 p-4 rounded-lg">
          {{ store.error }}
        </div>
      </div>
    </div>

    <!-- Input Form -->
    <div class="border-t bg-white p-4">
      <form @submit.prevent="handleSubmit" class="max-w-4xl mx-auto">
        <div class="flex gap-4">
          <input
            v-model="message"
            type="text"
            placeholder="Ask a question about your documents..."
            class="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            :disabled="store.isLoading"
          />
          <button
            type="submit"
            class="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50"
            :disabled="!message.trim() || store.isLoading"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useChatStore } from '~/stores/chat'

const store = useChatStore()
const message = ref('')

const handleSubmit = async () => {
  if (!message.value.trim() || store.isLoading) return
  await store.sendMessage(message.value.trim())
  message.value = ''
}

const handleFileUpload = async (event: Event) => {
  const input = event.target as HTMLInputElement
  if (!input.files?.length) return

  const file = input.files[0]
  const sourceId = prompt('Enter a unique identifier for this document:')
  if (!sourceId) return

  try {
    await store.uploadPDF(file, sourceId)
    alert('PDF uploaded successfully!')
  } catch (error) {
    alert('Failed to upload PDF. Please try again.')
  }
  
  // Reset input
  input.value = ''
}

const formatMessage = (content: string) => {
  // Convert newlines to <br> tags and escape HTML
  return content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
}
</script>

<style>
.prose {
  max-width: none;
}
</style> 