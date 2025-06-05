import { defineStore } from 'pinia'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  chunks?: any[]
}

interface ChatState {
  messages: Message[]
  isLoading: boolean
  error: string | null
}

export const useChatStore = defineStore('chat', {
  state: (): ChatState => ({
    messages: [],
    isLoading: false,
    error: null
  }),

  actions: {
    async sendMessage(content: string) {
      const config = useRuntimeConfig()
      this.isLoading = true
      this.error = null

      // Add user message
      const userMessage: Message = {
        id: Date.now().toString(),
        type: 'user',
        content,
        timestamp: new Date()
      }
      this.messages.push(userMessage)

      try {
        const response = await fetch(`${config.public.apiBase}/query/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: content,
            top_k: 10
          })
        })

        if (!response.ok) {
          throw new Error('Failed to get response')
        }

        const data = await response.json()
        
        // Add assistant message
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'assistant',
          content: data.answer,
          timestamp: new Date(),
          chunks: data.chunks
        }
        this.messages.push(assistantMessage)
      } catch (err) {
        this.error = err instanceof Error ? err.message : 'An error occurred'
      } finally {
        this.isLoading = false
      }
    },

    async uploadPDF(file: File, sourceId: string) {
      const config = useRuntimeConfig()
      this.isLoading = true
      this.error = null

      const formData = new FormData()
      formData.append('file', file)
      formData.append('source_id', sourceId)

      try {
        const response = await fetch(`${config.public.apiBase}/upload_pdf/`, {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          throw new Error('Failed to upload PDF')
        }

        const data = await response.json()
        return data
      } catch (err) {
        this.error = err instanceof Error ? err.message : 'An error occurred'
        throw err
      } finally {
        this.isLoading = false
      }
    },

    clearMessages() {
      this.messages = []
      this.error = null
    }
  }
}) 