/**
 * ESLint flat configuration for ruv-swarm
 * Comprehensive rules for TypeScript/ES modules, Node.js, and code quality
 * Using ESLint v9+ flat config format with simplified plugin loading
 */

// Import TypeScript parser for .ts files
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const tsParser = require('@typescript-eslint/parser');

export default [
  // Global ignores (replaces .eslintignore)
  {
    ignores: [
      'node_modules/**',
      'dist/**',
      'wasm/**',
      '*.wasm',
      '*.min.js',
      'coverage/**',
      '.nyc_output/**',
      'archive/**',
      'test-reports/**',
      '*.js.map',
      '*.d.ts.map',
      '*_bg.js',
      '*_bg.wasm',
      '*_bg.wasm.d.ts',
      'eslint-report.json',
      '*.log',
      '*.tmp',
      '*.temp',
      '.cache/**',
      'data/**',
      '*.db',
      '*.sqlite*',
      '../crates/**',
    ],
  },

  // JavaScript files configuration
  {
    files: ['**/*.js', '**/*.mjs', '**/*.cjs'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        process: 'readonly',
        Buffer: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        console: 'readonly',
        require: 'readonly',
        module: 'readonly',
        exports: 'readonly',
        global: 'readonly',
        setImmediate: 'readonly',
        clearImmediate: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        URL: 'readonly',
        URLSearchParams: 'readonly',
        fetch: 'readonly',
        WebAssembly: 'readonly',
        crypto: 'readonly',
        Blob: 'readonly',
        File: 'readonly',
        FormData: 'readonly',
        performance: 'readonly',
        jest: 'readonly',
        describe: 'readonly',
        it: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        beforeAll: 'readonly',
        afterAll: 'readonly',
      },
    },
    rules: {
      // Basic JavaScript rules
      'no-console': 'off', // Allow console in Node.js
      'no-debugger': 'error',
      'no-alert': 'error',
      'no-eval': 'error',
      'no-implied-eval': 'error',
      'no-new-func': 'error',
      'no-script-url': 'error',
      'no-var': 'error',
      'prefer-const': 'error',
      'prefer-arrow-callback': 'warn',
      'prefer-template': 'warn',
      'prefer-destructuring': 'warn',
      'prefer-spread': 'warn',
      'prefer-rest-params': 'warn',
      'object-shorthand': 'warn',

      // Error prevention
      'no-unused-vars': ['warn', {
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_',
      }],
      'no-undef': 'error',
      'no-redeclare': 'error',
      'no-unreachable': 'error',
      'no-constant-condition': 'error',
      'no-duplicate-case': 'error',
      'no-empty': 'error',
      'no-extra-boolean-cast': 'error',
      'no-func-assign': 'error',
      'no-invalid-regexp': 'error',
      'no-sparse-arrays': 'error',
      'no-unexpected-multiline': 'error',
      'use-isnan': 'error',
      'valid-typeof': 'error',

      // Best practices
      'curly': 'error',
      'dot-notation': 'warn',
      'eqeqeq': ['error', 'always'],
      'no-caller': 'error',
      'no-else-return': 'warn',
      'no-empty-function': 'warn',
      'no-eq-null': 'error',
      'no-global-assign': 'error',
      'no-implicit-coercion': 'warn',
      'no-implicit-globals': 'error',
      'no-lone-blocks': 'error',
      'no-loop-func': 'error',
      'no-multi-spaces': 'warn',
      'no-new': 'error',
      'no-new-wrappers': 'error',
      'no-return-assign': 'error',
      'no-self-assign': 'error',
      'no-self-compare': 'error',
      'no-sequences': 'error',
      'no-throw-literal': 'error',
      'no-unmodified-loop-condition': 'error',
      'no-unused-expressions': 'error',
      'no-useless-call': 'error',
      'no-useless-concat': 'error',
      'no-useless-return': 'error',
      'no-void': 'error',
      'no-with': 'error',
      'radix': 'error',
      'wrap-iife': 'error',
      'yoda': 'error',

      // Style consistency
      'array-bracket-spacing': ['warn', 'never'],
      'block-spacing': 'warn',
      'brace-style': ['warn', '1tbs'],
      'comma-dangle': ['warn', 'always-multiline'],
      'comma-spacing': 'warn',
      'comma-style': 'warn',
      'computed-property-spacing': 'warn',
      'func-call-spacing': 'warn',
      'indent': ['warn', 2],
      'key-spacing': 'warn',
      'keyword-spacing': 'warn',
      'no-multiple-empty-lines': ['warn', { max: 2 }],
      'no-trailing-spaces': 'warn',
      'object-curly-spacing': ['warn', 'always'],
      'quotes': ['warn', 'single', { avoidEscape: true }],
      'semi': ['warn', 'always'],
      'semi-spacing': 'warn',
      'space-before-blocks': 'warn',
      'space-before-function-paren': ['warn', 'never'],
      'space-in-parens': 'warn',
      'space-infix-ops': 'warn',
      'space-unary-ops': 'warn',
    },
  },

  // TypeScript files configuration
  {
    files: ['**/*.ts', '**/*.tsx'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 2022,
        sourceType: 'module',
        project: false, // Disable project to avoid TypeScript config issues
      },
      globals: {
        process: 'readonly',
        Buffer: 'readonly',
        __dirname: 'readonly',
        __filename: 'readonly',
        console: 'readonly',
        require: 'readonly',
        module: 'readonly',
        exports: 'readonly',
        global: 'readonly',
        setImmediate: 'readonly',
        clearImmediate: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        URL: 'readonly',
        URLSearchParams: 'readonly',
        fetch: 'readonly',
        WebAssembly: 'readonly',
        crypto: 'readonly',
        Blob: 'readonly',
        File: 'readonly',
        FormData: 'readonly',
        performance: 'readonly',
        jest: 'readonly',
        describe: 'readonly',
        it: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        beforeAll: 'readonly',
        afterAll: 'readonly',
      },
    },
    rules: {
      // Same rules as JavaScript but with TypeScript awareness
      'no-console': 'off',
      'no-debugger': 'error',
      'no-eval': 'error',
      'no-var': 'error',
      'prefer-const': 'error',
      'no-undef': 'off', // TypeScript handles this
      'no-unused-vars': 'off', // TypeScript handles this better
      'no-redeclare': 'off', // TypeScript handles this
      
      // Style rules (relaxed for TypeScript)
      'indent': ['warn', 2],
      'quotes': ['warn', 'single', { avoidEscape: true }],
      'semi': ['warn', 'always'],
      'comma-dangle': ['warn', 'always-multiline'],
      'object-curly-spacing': ['warn', 'always'],
    },
  },

  // Test files - more relaxed rules
  {
    files: ['**/*.test.js', '**/*.test.ts', '**/*.spec.js', '**/*.spec.ts', '**/test/**/*'],
    languageOptions: {
      parser: tsParser, // Use TypeScript parser for .ts test files too
      parserOptions: {
        ecmaVersion: 2022,
        sourceType: 'module',
        project: false, // Disable project for test files
      },
      globals: {
        jest: 'readonly',
        describe: 'readonly',
        it: 'readonly',
        expect: 'readonly',
        beforeEach: 'readonly',
        afterEach: 'readonly',
        beforeAll: 'readonly',
        afterAll: 'readonly',
        test: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        URL: 'readonly',
        fetch: 'readonly',
        WebAssembly: 'readonly',
        crypto: 'readonly',
        process: 'readonly',
        Buffer: 'readonly',
        console: 'readonly',
      },
    },
    rules: {
      'no-console': 'off',
      'prefer-destructuring': 'off',
      'no-unused-expressions': 'off', // For expect assertions
      'no-empty-function': 'off', // For test stubs
      'no-undef': 'off', // TypeScript/Jest handles this
      'no-unused-vars': 'off', // Test files may have unused vars
      'radix': 'warn', // More lenient for tests
      'no-loop-func': 'warn', // More lenient for tests
      'no-return-assign': 'warn', // More lenient for tests
    },
  },

  // Configuration files - very relaxed rules
  {
    files: ['**/*.config.js', '**/*.config.ts', '**/config/**/*', 'eslint.config.js'],
    rules: {
      'no-console': 'off',
      'prefer-destructuring': 'off',
    },
  },

  // Build and script files
  {
    files: ['**/scripts/**/*', '**/build/**/*', '**/tools/**/*', '**/bin/**/*'],
    rules: {
      'no-console': 'off',
      'prefer-destructuring': 'off',
    },
  },

  // WebAssembly related files - more flexible
  {
    files: ['**/wasm/**/*', '**/*wasm*', '**/*simd*'],
    rules: {
      'prefer-rest-params': 'off',
      'no-unused-vars': 'off', // WASM bindings may have unused exports
      'no-empty-function': 'off', // WASM may have empty method stubs
    },
  },

  // Neural network and AI files - more flexible
  {
    files: ['**/neural*', '**/agent*', '**/swarm*'],
    rules: {
      'prefer-destructuring': 'off',
      'no-unused-vars': 'warn', // AI code may have experimental variables
    },
  },
];