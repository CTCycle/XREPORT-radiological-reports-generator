// @ts-check
const eslint = require('@eslint/js');
const { defineConfig } = require('eslint/config');
const tseslint = require('typescript-eslint');
const angular = require('angular-eslint');

module.exports = defineConfig([
  {
    ignores: ['projects/shared/src/app/types/generated/**/types.gen.ts'],
  },
  {
    // Applications may only share code through projects/shared (@xreport/shared).
    files: ['projects/studio/**/*.ts'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [{ group: ['**/inference/src/**', '../inference/**', '../../inference/**'], message: 'Studio must not import XREPORT Inference code; move shared code to projects/shared.' }] }],
    },
  },
  {
    files: ['projects/inference/**/*.ts'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [{ group: ['**/studio/src/**', '../studio/**', '../../studio/**'], message: 'XREPORT Inference must not import Studio code; move shared code to projects/shared.' }] }],
    },
  },
  {
    files: ['projects/shared/**/*.ts'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [{ group: ['**/studio/src/**', '**/inference/src/**'], message: 'Shared code must not depend on an application.' }] }],
    },
  },
  {
    files: ['**/*.ts'],
    extends: [
      eslint.configs.recommended,
      tseslint.configs.recommended,
      tseslint.configs.stylistic,
      angular.configs.tsRecommended,
    ],
    processor: angular.processInlineTemplates,
    rules: {
      '@angular-eslint/directive-selector': [
        'error',
        {
          type: 'attribute',
          prefix: 'app',
          style: 'camelCase',
        },
      ],
      '@angular-eslint/component-selector': [
        'error',
        {
          type: 'element',
          prefix: 'app',
          style: 'kebab-case',
        },
      ],
      '@angular-eslint/template/click-events-have-key-events': 'off',
      '@angular-eslint/template/prefer-control-flow': 'off',
    },
  },
  {
    files: ['**/*.html'],
    extends: [angular.configs.templateRecommended, angular.configs.templateAccessibility],
    rules: {
      '@angular-eslint/template/click-events-have-key-events': 'off',
      '@angular-eslint/template/prefer-control-flow': 'off',
    },
  },
  {
    files: ['**/src/app/pages/dataset.page.ts'],
    rules: {
      // The modal backdrop uses pointer dismissal while the explicit close/cancel controls remain keyboard accessible.
      '@angular-eslint/template/click-events-have-key-events': 'off',
      '@angular-eslint/template/prefer-control-flow': 'off',
    },
  },
]);
