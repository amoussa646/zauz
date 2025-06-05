import { defineStore } from 'pinia'
import { useRuntimeConfig } from '#imports'
import { ref } from 'vue'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: number
  chunks?: Array<{
    source: string
    page: number
    chunk: string
  }>
  relevantImages?: Array<{
    filename: string
  }>
}

interface ImageMetadata {
  imageId: string
  description: string
  tags: string[]
  contextKeywords: string[]
}

interface ChatState {
  messages: Message[]
  loading: boolean
  error: string | null
}

export const useChatStore = defineStore('chat', () => {
  const config = useRuntimeConfig()
  const messages = ref<Message[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  const sendMessage = async (content: string) => {
    const messageId = Date.now().toString()
    messages.value.push({
      id: messageId,
      type: 'user',
      content,
      timestamp: Date.now()
    })

    loading.value = true
    error.value = null

    try {
      const response = await fetch(`${config.public.apiBase}/query_docs/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          question: content,
          include_images: true 
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      messages.value.push({
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.answer,
        timestamp: Date.now(),
        chunks: data.chunks,
        relevantImages: data.relevant_images.map((img: { filename: string }) => ({
          filename: img.filename
        }))
      })
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'An error occurred'
      console.error('Error sending message:', err)
    } finally {
      loading.value = false
    }
  }

  const uploadImage = async (file: File, metadata: ImageMetadata) => {
    loading.value = true
    error.value = null

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('image_id', metadata.imageId)
      formData.append('description', metadata.description)
      formData.append('tags', JSON.stringify(metadata.tags))
      formData.append('context_keywords', JSON.stringify(metadata.contextKeywords))

      const response = await fetch(`${config.public.apiBase}/upload_image/`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Failed to upload image')
      }

      const data = await response.json()
      return data
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Failed to upload image'
      throw err
    } finally {
      loading.value = false
    }
  }

  const getImageUrl = (filename: string) => {
    return `${config.public.apiBase}/images/${filename}`
  }

  const clearMessages = () => {
    messages.value = []
    error.value = null
  }

  return {
    messages,
    loading,
    error,
    sendMessage,
    uploadImage,
    getImageUrl,
    clearMessages
  }
}) 