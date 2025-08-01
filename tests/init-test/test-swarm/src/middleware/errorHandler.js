const { logger } = require('../utils/logger');

const errorHandler = (err, req, res, next) => {
  logger.error({
    message: err.message,
    stack: err.stack,
    method: req.method,
    url: req.url,
    ip: req.ip,
  });

  const status = err.status || 500;
  const message = process.env.NODE_ENV === 'production'
    ? 'Internal Server Error'
    : err.message;

  res.status(status).json({
    error: {
      message,
      status,
      ...(process.env.NODE_ENV !== 'production' && { stack: err.stack }),
    },
  });
};

module.exports = { errorHandler };
