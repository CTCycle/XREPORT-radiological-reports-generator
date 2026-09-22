'use strict';

const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const DEFAULT_DIST_ROOT = path.resolve(__dirname, '..', 'dist', 'client-angular', 'browser');
const HOP_BY_HOP_HEADERS = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
]);
const CONTENT_TYPES = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.gif', 'image/gif'],
  ['.html', 'text/html; charset=utf-8'],
  ['.ico', 'image/x-icon'],
  ['.jpeg', 'image/jpeg'],
  ['.jpg', 'image/jpeg'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.map', 'application/json; charset=utf-8'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
  ['.ttf', 'font/ttf'],
  ['.woff', 'font/woff'],
  ['.woff2', 'font/woff2'],
]);

function normalizeApiBase(value) {
  const text = String(value || '/api').trim();
  if (!text || text.includes('://') || text.startsWith('//')) return '/api';
  const withLeadingSlash = text.startsWith('/') ? text : `/${text}`;
  if (withLeadingSlash === '/') return '/';
  return withLeadingSlash.replace(/\/+$/, '');
}

function parsePort(value, label) {
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error(`${label} must be an integer between 1 and 65535.`);
  }
  return port;
}

function readOption(args, index, name) {
  const value = args[index];
  if (value === name) {
    if (index + 1 >= args.length) throw new Error(`${name} requires a value.`);
    return { value: args[index + 1], nextIndex: index + 1 };
  }
  const prefix = `${name}=`;
  if (value.startsWith(prefix)) return { value: value.slice(prefix.length), nextIndex: index };
  return null;
}

function parseArgs(args = process.argv.slice(2), environment = process.env) {
  const options = {
    host: environment.UI_HOST || '127.0.0.1',
    port: parsePort(environment.UI_PORT || '8003', 'UI_PORT'),
    apiBaseUrl: normalizeApiBase(environment.UI_API_BASE_URL || '/api'),
    backendHost: environment.FASTAPI_HOST || '127.0.0.1',
    backendPort: parsePort(environment.FASTAPI_PORT || '5003', 'FASTAPI_PORT'),
    distRoot: DEFAULT_DIST_ROOT,
  };

  for (let index = 0; index < args.length; index += 1) {
    let option = readOption(args, index, '--host');
    if (option) {
      options.host = option.value;
      index = option.nextIndex;
      continue;
    }
    option = readOption(args, index, '--port');
    if (option) {
      options.port = parsePort(option.value, 'port');
      index = option.nextIndex;
      continue;
    }
    option = readOption(args, index, '--dist');
    if (option) {
      options.distRoot = path.resolve(option.value);
      index = option.nextIndex;
      continue;
    }
    option = readOption(args, index, '--api-base-url');
    if (option) {
      options.apiBaseUrl = normalizeApiBase(option.value);
      index = option.nextIndex;
      continue;
    }
    option = readOption(args, index, '--backend-host');
    if (option) {
      options.backendHost = option.value;
      index = option.nextIndex;
      continue;
    }
    option = readOption(args, index, '--backend-port');
    if (option) {
      options.backendPort = parsePort(option.value, 'backend-port');
      index = option.nextIndex;
      continue;
    }
    throw new Error(`Unsupported preview option: ${args[index]}`);
  }

  options.distRoot = path.resolve(options.distRoot);
  return options;
}

function getRequestUrl(request) {
  return new URL(request.url || '/', 'http://xreport.local');
}

function isApiRequest(requestUrl, apiBaseUrl) {
  return apiBaseUrl === '/'
    ? true
    : requestUrl.pathname === apiBaseUrl || requestUrl.pathname.startsWith(`${apiBaseUrl}/`);
}

function isWithinRoot(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === '' || (
    relative !== '..'
    && !relative.startsWith(`..${path.sep}`)
    && !path.isAbsolute(relative)
  );
}

function resolveStaticPath(distRoot, requestPath) {
  let decodedPath;
  try {
    decodedPath = decodeURIComponent(requestPath);
  } catch {
    return null;
  }
  if (decodedPath.includes('\0')) return null;

  const normalizedPath = decodedPath.replaceAll('\\', '/');
  const root = path.resolve(distRoot);
  const candidate = path.resolve(root, `.${normalizedPath}`);
  if (!isWithinRoot(root, candidate)) return null;
  return candidate;
}

function sendJson(response, statusCode, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store',
  });
  response.end(body);
}

function contentTypeFor(filePath) {
  return CONTENT_TYPES.get(path.extname(filePath).toLowerCase()) || 'application/octet-stream';
}

function sendStaticFile(request, response, filePath) {
  let stats;
  try {
    stats = fs.statSync(filePath);
  } catch {
    sendJson(response, 404, { detail: 'Static file not found.' });
    return;
  }
  if (!stats.isFile()) {
    sendJson(response, 404, { detail: 'Static file not found.' });
    return;
  }

  response.writeHead(200, {
    'Content-Type': contentTypeFor(filePath),
    'Content-Length': stats.size,
  });
  if (request.method === 'HEAD') {
    response.end();
    return;
  }

  const stream = fs.createReadStream(filePath);
  stream.on('error', () => {
    if (!response.headersSent) sendJson(response, 500, { detail: 'Unable to read static file.' });
    else response.destroy();
  });
  stream.pipe(response);
}

function serveStatic(request, response, options) {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    sendJson(response, 405, { detail: 'Method not allowed.' });
    return;
  }

  const rawPath = String(request.url || '/').split('?', 1)[0] || '/';
  const candidate = resolveStaticPath(options.distRoot, rawPath);
  if (candidate === null) {
    sendJson(response, 403, { detail: 'The requested path is outside the frontend bundle.' });
    return;
  }

  let filePath = candidate;
  try {
    if (!fs.statSync(filePath).isFile()) filePath = path.join(options.distRoot, 'index.html');
  } catch {
    filePath = path.join(options.distRoot, 'index.html');
  }

  let resolvedFilePath;
  try {
    resolvedFilePath = fs.realpathSync(filePath);
  } catch {
    sendJson(response, 404, { detail: 'Static file not found.' });
    return;
  }
  if (!isWithinRoot(options.realDistRoot, resolvedFilePath)) {
    sendJson(response, 403, { detail: 'The requested path is outside the frontend bundle.' });
    return;
  }
  sendStaticFile(request, response, resolvedFilePath);
}

function forwardResponse(proxyResponse, clientResponse) {
  const headers = {};
  for (const [name, value] of Object.entries(proxyResponse.headers)) {
    if (!HOP_BY_HOP_HEADERS.has(name.toLowerCase()) && value !== undefined) headers[name] = value;
  }
  clientResponse.writeHead(proxyResponse.statusCode || 502, headers);
  proxyResponse.pipe(clientResponse);
}

function proxyApiRequest(request, response, options) {
  const requestUrl = getRequestUrl(request);
  const headers = {};
  for (const [name, value] of Object.entries(request.headers)) {
    if (!HOP_BY_HOP_HEADERS.has(name.toLowerCase()) && name.toLowerCase() !== 'host') headers[name] = value;
  }
  headers.host = `${options.backendHost}:${options.backendPort}`;

  const proxyRequest = http.request(
    {
      hostname: options.backendHost,
      port: options.backendPort,
      method: request.method,
      path: `${requestUrl.pathname}${requestUrl.search}`,
      headers,
    },
    (proxyResponse) => forwardResponse(proxyResponse, response),
  );

  let responded = false;
  proxyRequest.once('response', () => {
    responded = true;
  });
  proxyRequest.once('error', (error) => {
    if (responded || response.headersSent) {
      response.destroy();
      return;
    }
    sendJson(response, 502, {
      detail: 'The XREPORT backend is not available yet.',
      error: error.code || 'backend_unavailable',
    });
  });
  request.once('aborted', () => proxyRequest.destroy());
  request.pipe(proxyRequest);
}

function createBuiltServer(options) {
  const resolvedOptions = {
    ...options,
    distRoot: path.resolve(options.distRoot),
    apiBaseUrl: normalizeApiBase(options.apiBaseUrl),
    backendPort: parsePort(options.backendPort, 'backendPort'),
  };
  resolvedOptions.realDistRoot = fs.realpathSync(resolvedOptions.distRoot);
  const indexPath = path.join(resolvedOptions.distRoot, 'index.html');
  if (!fs.existsSync(indexPath)) {
    throw new Error(`Built frontend index was not found: ${indexPath}. Run npm run build first.`);
  }

  return http.createServer((request, response) => {
    try {
      const requestUrl = getRequestUrl(request);
      if (isApiRequest(requestUrl, resolvedOptions.apiBaseUrl)) {
        proxyApiRequest(request, response, resolvedOptions);
      } else {
        serveStatic(request, response, resolvedOptions);
      }
    } catch (error) {
      if (!response.headersSent) sendJson(response, 500, { detail: 'Frontend server request failed.' });
      else response.destroy();
    }
  });
}

function start() {
  const options = parseArgs();
  const server = createBuiltServer(options);
  server.listen(options.port, options.host, () => {
    console.log(
      `XREPORT built frontend listening on http://${options.host}:${options.port} (backend ${options.backendHost}:${options.backendPort})`,
    );
  });
  server.on('error', (error) => {
    console.error(`XREPORT built frontend failed to listen: ${error.message}`);
    process.exitCode = 1;
  });
}

if (require.main === module) start();

module.exports = {
  createBuiltServer,
  normalizeApiBase,
  parseArgs,
  resolveStaticPath,
};
