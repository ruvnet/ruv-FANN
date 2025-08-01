import { nodeResolve } from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import wasm from '@rollup/plugin-wasm';
import { terser } from 'rollup-plugin-terser';

export default [
  // Browser build
  {
    input: 'src/index.js',
    output: {
      file: 'dist/ruv-swarm.browser.js',
      format: 'es',
      sourcemap: true,
    },
    plugins: [
      nodeResolve({
        browser: true,
        preferBuiltins: false,
      }),
      commonjs(),
      wasm({
        sync: ['*'],
        fileName: '[name]-[hash][extname]',
        publicPath: '/wasm/',
      }),
      terser(),
    ],
    external: [],
  },
  // Node.js build
  {
    input: 'src/index.js',
    output: {
      file: 'dist/ruv-swarm.node.js',
      format: 'cjs',
      sourcemap: true,
    },
    plugins: [
      nodeResolve({
        preferBuiltins: true,
      }),
      commonjs(),
      wasm({
        sync: ['*'],
        targetEnv: 'node',
      }),
    ],
    external: ['fs', 'path', 'worker_threads'],
  },
  // UMD build for CDN
  {
    input: 'src/index.js',
    output: {
      file: 'dist/ruv-swarm.umd.js',
      format: 'umd',
      name: 'RuvSwarm',
      sourcemap: true,
    },
    plugins: [
      nodeResolve({
        browser: true,
        preferBuiltins: false,
      }),
      commonjs(),
      wasm({
        sync: ['*'],
        fileName: '[name][extname]',
        publicPath: '/wasm/',
      }),
      terser(),
    ],
  },
];
