const normalizeApiBase = (value) => {
  if (!value) return '/api';
  const trimmed = String(value).trim();
  if (!trimmed || trimmed.includes('://') || trimmed.startsWith('//')) return '/api';
  const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
  return withLeadingSlash.length > 1 && withLeadingSlash.endsWith('/')
    ? withLeadingSlash.slice(0, -1)
    : withLeadingSlash;
};

const apiBase = normalizeApiBase(process.env.UI_API_BASE_URL || '/api');
const apiHost = process.env.FASTAPI_HOST || '127.0.0.1';
const apiPort = process.env.FASTAPI_PORT || '5003';

module.exports = {
  [`${apiBase}/**`]: {
    target: `http://${apiHost}:${apiPort}`,
    secure: false,
    changeOrigin: true,
  },
};
