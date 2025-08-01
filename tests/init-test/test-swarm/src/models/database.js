const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const { logger } = require('../utils/logger');

const dbPath = process.env.DATABASE_URL || './database.sqlite';

const db = new sqlite3.Database(dbPath, (err) => {
  if (err) {
    logger.error('Error opening database:', err);
  } else {
    logger.info('Connected to SQLite database');
  }
});

const initializeDatabase = () => {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      // Users table
      db.run(`
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE NOT NULL,
          email TEXT UNIQUE NOT NULL,
          password TEXT NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
      `, (err) => {
        if (err) {
          logger.error('Error creating users table:', err);
          reject(err);
        }
      });

      // Sessions table
      db.run(`
        CREATE TABLE IF NOT EXISTS sessions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          token TEXT UNIQUE NOT NULL,
          expires_at DATETIME NOT NULL,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users (id)
        )
      `, (err) => {
        if (err) {
          logger.error('Error creating sessions table:', err);
          reject(err);
        }
      });

      // API logs table for monitoring
      db.run(`
        CREATE TABLE IF NOT EXISTS api_logs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          method TEXT NOT NULL,
          path TEXT NOT NULL,
          status_code INTEGER NOT NULL,
          response_time INTEGER NOT NULL,
          user_id INTEGER,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users (id)
        )
      `, (err) => {
        if (err) {
          logger.error('Error creating api_logs table:', err);
          reject(err);
        } else {
          logger.info('Database tables initialized');
          resolve();
        }
      });
    });
  });
};

module.exports = { db, initializeDatabase };
