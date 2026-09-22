'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { after, before, test } = require('node:test');

const { createBuiltServer } = require('./serve-built.cjs');

let distRoot;
let temporaryRoot;
let backend;
let backendPort;
let frontend;
let frontendPort;
let symlinkAvailable = false;

function listen(server) {
  return new Promise((resolve, reject) => {
    const onError = (error) => {
      server.off('listening', onListening);
      reject(error);
    };
    const onListening = () => {
      server.off('error', onError);
      resolve(server.address().port);
    };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(0, '127.0.0.1');
  });
}

function close(server) {
  return new Promise((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
}

function request(port, options = {}, body = '') {
  return new Promise((resolve, reject) => {
    const requestOptions = {
      hostname: '127.0.0.1',
      port,
      path: '/',
      method: 'GET',
      ...options,
    };
    const clientRequest = http.request(requestOptions, (response) => {
      const chunks = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => {
        resolve({
          statusCode: response.statusCode,
          headers: response.headers,
          body: Buffer.concat(chunks).toString('utf8'),
        });
      });
    });
    clientRequest.on('error', reject);
    if (body) clientRequest.write(body);
    clientRequest.end();
  });
}

before(async () => {
  temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'xreport-built-server-'));
  distRoot = path.join(temporaryRoot, 'browser');
  fs.mkdirSync(distRoot);
  fs.writeFileSync(path.join(distRoot, 'index.html'), '<html><body>shell</body></html>');
  fs.mkdirSync(path.join(distRoot, 'assets'));
  fs.writeFileSync(path.join(distRoot, 'assets', 'main.abc.js'), 'console.log("asset");');
  fs.writeFileSync(path.join(distRoot, 'secret.txt'), 'not outside');
  const outsideFile = path.join(temporaryRoot, 'outside-secret.txt');
  fs.writeFileSync(outsideFile, 'must not be served');
  try {
    fs.symlinkSync(outsideFile, path.join(distRoot, 'outside-link.txt'));
    symlinkAvailable = true;
  } catch (error) {
    if (!['EPERM', 'EACCES', 'ENOTSUP'].includes(error.code)) throw error;
  }

  backend = http.createServer((request, response) => {
    if (request.url === '/api/health') {
      response.writeHead(200, { 'Content-Type': 'application/json' });
      response.end('{"status":"ok"}');
      return;
    }
    if (request.url === '/api/status') {
      response.writeHead(418, { 'Content-Type': 'application/json' });
      response.end('{"status":"teapot"}');
      return;
    }
    if (request.url === '/api/echo' && request.method === 'POST') {
      const chunks = [];
      request.on('data', (chunk) => chunks.push(chunk));
      request.on('end', () => {
        response.writeHead(201, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify({ body: Buffer.concat(chunks).toString('utf8') }));
      });
      return;
    }
    response.writeHead(404);
    response.end();
  });
  backendPort = await listen(backend);
  frontend = createBuiltServer({
    distRoot,
    apiBaseUrl: '/api',
    backendHost: '127.0.0.1',
    backendPort,
  });
  frontendPort = await listen(frontend);
});

after(async () => {
  await close(frontend);
  await close(backend);
  fs.rmSync(temporaryRoot, { recursive: true, force: true });
});

test('serves index.html at the root', async () => {
  const response = await request(frontendPort);
  assert.equal(response.statusCode, 200);
  assert.match(response.body, /shell/);
});

test('serves hashed assets directly', async () => {
  const response = await request(frontendPort, { path: '/assets/main.abc.js' });
  assert.equal(response.statusCode, 200);
  assert.match(response.body, /console\.log/);
  assert.match(response.headers['content-type'], /javascript/);
});

test('falls back to index.html for Angular client routes', async () => {
  const response = await request(frontendPort, { path: '/inference/123' });
  assert.equal(response.statusCode, 200);
  assert.match(response.body, /shell/);
});

test('proxies API health and preserves the backend status', async () => {
  const response = await request(frontendPort, { path: '/api/health' });
  assert.equal(response.statusCode, 200);
  assert.equal(JSON.parse(response.body).status, 'ok');
});

test('forwards non-GET request bodies and status codes', async () => {
  const response = await request(
    frontendPort,
    { path: '/api/echo', method: 'POST', headers: { 'Content-Type': 'application/json' } },
    '{"value":42}',
  );
  assert.equal(response.statusCode, 201);
  assert.equal(JSON.parse(response.body).body, '{"value":42}');
});

test('propagates non-success backend responses', async () => {
  const response = await request(frontendPort, { path: '/api/status' });
  assert.equal(response.statusCode, 418);
});

test('returns controlled 502 responses while the backend is unavailable', async () => {
  await close(backend);
  const response = await request(frontendPort, { path: '/api/health' });
  assert.equal(response.statusCode, 502);
  const body = JSON.parse(response.body);
  assert.equal(body.detail, 'The XREPORT backend is not available yet.');
  assert.ok(['ECONNREFUSED', 'ECONNRESET'].includes(body.error));

  backend = http.createServer((request, response) => {
    response.writeHead(200);
    response.end();
  });
  backendPort = await listen(backend);
});

test('rejects traversal attempts without leaving the dist root', async () => {
  const response = await request(frontendPort, { path: '/%2e%2e/secret.txt' });
  assert.equal(response.statusCode, 403);
  assert.doesNotMatch(response.body, /not outside/);
});

test('does not serve symlinks whose targets leave the dist root', { skip: !symlinkAvailable }, async () => {
  const response = await request(frontendPort, { path: '/outside-link.txt' });
  assert.equal(response.statusCode, 403);
  assert.doesNotMatch(response.body, /must not be served/);
});
