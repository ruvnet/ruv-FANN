const bcrypt = require('bcrypt');
const { db } = require('./database');

class User {
  static async create({ username, email, password }) {
    const hashedPassword = await bcrypt.hash(password, 10);

    return new Promise((resolve, reject) => {
      const query = `
        INSERT INTO users (username, email, password)
        VALUES (?, ?, ?)
      `;

      db.run(query, [username, email, hashedPassword], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve({ id: this.lastID, username, email });
        }
      });
    });
  }

  static async findAll() {
    return new Promise((resolve, reject) => {
      const query = 'SELECT id, username, email, created_at FROM users ORDER BY created_at DESC';

      db.all(query, [], (err, rows) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows);
        }
      });
    });
  }

  static async findByEmail(email) {
    return new Promise((resolve, reject) => {
      const query = 'SELECT * FROM users WHERE email = ?';

      db.get(query, [email], (err, row) => {
        if (err) {
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  }

  static async findById(id) {
    return new Promise((resolve, reject) => {
      const query = 'SELECT id, username, email, created_at FROM users WHERE id = ?';

      db.get(query, [id], (err, row) => {
        if (err) {
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  }

  static async findByIdWithPassword(id) {
    return new Promise((resolve, reject) => {
      const query = 'SELECT * FROM users WHERE id = ?';

      db.get(query, [id], (err, row) => {
        if (err) {
          reject(err);
        } else {
          resolve(row);
        }
      });
    });
  }

  static async update(id, updates) {
    return new Promise((resolve, reject) => {
      const fields = [];
      const values = [];

      Object.keys(updates).forEach(key => {
        fields.push(`${key} = ?`);
        values.push(updates[key]);
      });

      values.push(id);

      const query = `
        UPDATE users 
        SET ${fields.join(', ')}, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
      `;

      db.run(query, values, function(err) {
        if (err) {
          reject(err);
        } else {
          User.findById(id).then(resolve).catch(reject);
        }
      });
    });
  }

  static async updatePassword(id, hashedPassword) {
    return new Promise((resolve, reject) => {
      const query = `
        UPDATE users 
        SET password = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
      `;

      db.run(query, [hashedPassword, id], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve({ affected: this.changes });
        }
      });
    });
  }

  static async delete(id) {
    return new Promise((resolve, reject) => {
      const query = 'DELETE FROM users WHERE id = ?';

      db.run(query, [id], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve({ deleted: this.changes > 0 });
        }
      });
    });
  }

  static async verifyPassword(password, hashedPassword) {
    return bcrypt.compare(password, hashedPassword);
  }
}

module.exports = User;
