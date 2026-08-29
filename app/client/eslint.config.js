// @ts-check
const eslint = require('@eslint/js');
const { defineConfig } = require('eslint/config');
const tseslint = require('typescript-eslint');
const angular = require('angular-eslint');

module.exports = defineConfig([
  {
    ignores: ['src/app/types/api.generated.ts'],
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
