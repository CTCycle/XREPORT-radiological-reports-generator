import path from 'node:path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const normalizeApiBase = (value) => {
    if (!value) {
        return '/api'
    }

    const trimmed = value.trim()
    if (!trimmed || trimmed.includes('://') || trimmed.startsWith('//')) {
        return '/api'
    }

    const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
    if (withLeadingSlash.length > 1 && withLeadingSlash.endsWith('/')) {
        return withLeadingSlash.slice(0, -1)
    }

    return withLeadingSlash
}

const buildProxy = (apiBase, apiTarget) => {
    const wsTarget = apiTarget.replace('http', 'ws')

    return {
        [`${apiBase}/training/ws`]: {
            target: wsTarget,
            ws: true,
            changeOrigin: true,
        },
        [`${apiBase}/inference/ws`]: {
            target: wsTarget,
            ws: true,
            changeOrigin: true,
        },
        [apiBase]: {
            target: apiTarget,
            changeOrigin: true,
        },
    }
}

export default defineConfig(({ mode }) => {
    const envDir = path.resolve(__dirname, '../settings')
    const clientEnv = loadEnv(mode, __dirname, '')
    const settingsEnv = loadEnv(mode, envDir, '')
    const env = { ...process.env, ...clientEnv, ...settingsEnv }

    const apiHost = env.FASTAPI_HOST || '127.0.0.1'
    const apiPort = env.FASTAPI_PORT || '8000'
    const apiTarget = `http://${apiHost}:${apiPort}`

    const uiHost = env.UI_HOST || '127.0.0.1'
    const uiPort = Number.parseInt(env.UI_PORT || '5173', 10)
    const apiBase = normalizeApiBase(env.VITE_API_BASE_URL || '/api')

    return {
        envDir,
        plugins: [react()],
        server: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: buildProxy(apiBase, apiTarget),
        },
        preview: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: buildProxy(apiBase, apiTarget),
        },
    }
})
