# RAG Chat Interface Frontend

A modern Vue.js/Nuxt.js frontend for interacting with the RAG pipeline. This interface allows users to upload PDF documents and ask questions about their content.

## Features

- PDF document upload
- Real-time chat interface
- Display of relevant document chunks with each answer
- Modern, responsive design
- Loading states and error handling

## Prerequisites

- Node.js 16.x or later
- npm or yarn
- Backend RAG API running (default: http://localhost:8000)

## Setup

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Create a `.env` file in the frontend directory:
```env
API_BASE=http://localhost:8000  # Change this if your backend runs on a different URL
```

3. Start the development server:
```bash
npm run dev
# or
yarn dev
```

The application will be available at http://localhost:3000

## Usage

1. Click the "Upload PDF" button to upload a document
2. Enter a unique identifier for the document when prompted
3. Wait for the upload to complete
4. Start asking questions about the uploaded documents
5. View the answers along with relevant document chunks
6. Use the "Clear Chat" button to start a new conversation

## Building for Production

To build the application for production:

```bash
npm run build
# or
yarn build
```

The built files will be in the `.output` directory.

## Development

- The main chat interface is in `components/ChatInterface.vue`
- State management is handled by Pinia in `stores/chat.ts`
- The application uses Tailwind CSS for styling
- API endpoints are configured in `nuxt.config.ts` 