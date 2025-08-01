const express = require('express');
const { db } = require('../models/database');
const os = require('os');

const router = express.Router();

// Health check endpoint
router.get('/', (req, res) => {
  const healthcheck = {
    uptime: process.uptime(),
    status: 'OK',
    timestamp: Date.now(),
    environment: process.env.NODE_ENV,
    memory: {
      used: process.memoryUsage(),
      free: os.freemem(),
      total: os.totalmem(),
    },
  };

  // Check database connection
  db.get('SELECT 1', (err) => {
    if (err) {
      healthcheck.status = 'ERROR';
      healthcheck.database = 'disconnected';
      res.status(503).json(healthcheck);
    } else {
      healthcheck.database = 'connected';
      res.json(healthcheck);
    }
  });
});

module.exports = router;
