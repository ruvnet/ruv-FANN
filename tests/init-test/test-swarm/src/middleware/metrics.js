const client = require('prom-client');
const { logger } = require('../utils/logger');

const register = new client.Registry();

// Default metrics
client.collectDefaultMetrics({ register });

// Custom metrics
const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
});

const httpRequestTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
});

register.registerMetric(httpRequestDuration);
register.registerMetric(httpRequestTotal);

const metricsMiddleware = (req, res, next) => {
  const start = Date.now();

  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    const route = req.route ? req.route.path : req.path;

    httpRequestDuration
      .labels(req.method, route, res.statusCode)
      .observe(duration);

    httpRequestTotal
      .labels(req.method, route, res.statusCode)
      .inc();
  });

  next();
};

const registerMetrics = (app) => {
  const metricsPort = process.env.METRICS_PORT || 9090;

  app.get('/metrics', (req, res) => {
    res.set('Content-Type', register.contentType);
    register.metrics().then(data => res.send(data));
  });

  logger.info(`Metrics endpoint available at /metrics`);
};

module.exports = { metricsMiddleware, registerMetrics };
