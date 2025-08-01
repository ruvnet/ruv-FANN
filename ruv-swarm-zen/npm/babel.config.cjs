/**
 * Babel Configuration for ruv-swarm
 * Handles ES modules, CommonJS compatibility, and modern JavaScript features
 */

module.exports = {
  presets: [
    [
      '@babel/preset-env',
      {
        targets: {
          node: '14',
        },
        modules: 'auto', // Let Babel decide based on the environment
        useBuiltIns: 'usage',
        corejs: 3,
      },
    ],
  ],

  plugins: [
    // Support for dynamic imports
    '@babel/plugin-syntax-dynamic-import',

    // Modern JavaScript features (use transform versions for latest features)
    '@babel/plugin-transform-optional-chaining',
    '@babel/plugin-transform-nullish-coalescing-operator',
    '@babel/plugin-transform-logical-assignment-operators',

    // Class properties
    '@babel/plugin-transform-class-properties',
    '@babel/plugin-transform-private-methods',

    // Object rest/spread
    '@babel/plugin-transform-object-rest-spread',

    // Async/await
    '@babel/plugin-transform-async-to-generator',

    // Export default from
    '@babel/plugin-transform-export-namespace-from',
  ],

  // Environment-specific configurations
  env: {
    test: {
      presets: [
        [
          '@babel/preset-env',
          {
            targets: {
              node: 'current',
            },
            modules: 'commonjs', // Use CommonJS for Jest
          },
        ],
      ],
    },

    production: {
      presets: [
        [
          '@babel/preset-env',
          {
            targets: {
              node: '14',
            },
            modules: false, // Keep ES modules for production
          },
        ],
      ],
    },

    development: {
      presets: [
        [
          '@babel/preset-env',
          {
            targets: {
              node: '14',
            },
            modules: false, // Keep ES modules for development
          },
        ],
      ],
    },
  },

  // Source maps for debugging
  sourceMaps: true,

  // Ignore certain files
  ignore: [
    'node_modules/**',
    'wasm/**',
    'target/**',
    '**/*.wasm',
  ],
};
