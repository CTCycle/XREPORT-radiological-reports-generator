import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: '../shared/openapi.json',
  output: {
    entryFile: false,
    path: 'src/app/types/generated',
  },
  plugins: ['@hey-api/typescript'],
});
