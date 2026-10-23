import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig([
  {
    input: '../shared/openapi.xreport.json',
    output: { entryFile: false, path: 'src/app/types/generated/xreport' },
    plugins: ['@hey-api/typescript'],
  },
  {
    input: '../shared/openapi.studio.json',
    output: { entryFile: false, path: 'src/app/types/generated/studio' },
    plugins: ['@hey-api/typescript'],
  },
]);
