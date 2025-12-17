import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
const apiHost = process.env.FASTAPI_HOST || '127.0.0.1'
const apiPort = process.env.FASTAPI_PORT || '8000'
const apiTarget = `http://${apiHost}:${apiPort}`


export default defineConfig({
    plugins: [react()],
    server: {
        host: '127.0.0.1',
        port: 7861,
        strictPort: false,
        proxy: {
            '/api': {
                target: apiTarget,
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api/, ''),
            },
        },
    },
    preview: {
        host: '127.0.0.1',
        port: 7861,
        strictPort: false,
        proxy: {
            '/api': {
                target: apiTarget,
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/api/, ''),
            },
        },
    },
})
