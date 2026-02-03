import path from 'node:path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const normalizeApiBase = (value: string) => {
    if (!value) {
        return '/api'
    }

    const withLeadingSlash = value.startsWith('/') ? value : `/${value}`
    if (withLeadingSlash.length > 1 && withLeadingSlash.endsWith('/')) {
        return withLeadingSlash.slice(0, -1)
    }

    return withLeadingSlash
}

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

const buildProxy = (apiBase: string, apiTarget: string) => {
    const wsTarget = apiTarget.replace('http', 'ws')
    const baseRegex = new RegExp(`^${escapeRegExp(apiBase)}`)

    return {
        [`${apiBase}/training/ws`]: {
            target: wsTarget,
            ws: true,
            changeOrigin: true,
            rewrite: (proxyPath: string) => proxyPath.replace(baseRegex, ''),
        },
        [`${apiBase}/inference/ws`]: {
            target: wsTarget,
            ws: true,
            changeOrigin: true,
            rewrite: (proxyPath: string) => proxyPath.replace(baseRegex, ''),
        },
        [apiBase]: {
            target: apiTarget,
            changeOrigin: true,
            rewrite: (proxyPath: string) => proxyPath.replace(baseRegex, ''),
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
