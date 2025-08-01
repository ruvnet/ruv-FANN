const path = require('path');
const webpack = require('webpack');

module.exports = {
  mode: 'production',
  entry: './src/index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'ruv-swarm.js',
    library: {
      name: 'RuvSwarm',
      type: 'umd',
      export: 'default',
    },
    globalObject: 'this',
  },
  module: {
    rules: [
      {
        test: /\.wasm$/,
        type: 'asset/resource',
        generator: {
          filename: 'wasm/[name][ext]',
        },
      },
    ],
  },
  experiments: {
    asyncWebAssembly: true,
    syncWebAssembly: true,
  },
  resolve: {
    fallback: {
      fs: false,
      path: false,
      crypto: false,
      buffer: false,
      util: false,
      stream: false,
      assert: false,
    },
  },
  plugins: [
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify('production'),
    }),
    new webpack.IgnorePlugin({
      resourceRegExp: /^fs$/,
    }),
  ],
  optimization: {
    minimize: true,
    usedExports: true,
    sideEffects: false,
  },
  performance: {
    hints: 'warning',
    maxAssetSize: 512000,
    maxEntrypointSize: 512000,
  },
};
