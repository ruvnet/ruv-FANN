/** Bound compiler output/lifetime and terminate the compiler process tree. */
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';

export function runBuildProcess(executable, args, options = {}) {
  const { timeout = 600000, maximumOutput = 8 * 1024 * 1024, output = process.stdout, errorOutput = process.stderr,
    ...spawnOptions } = options;
  if (!Number.isSafeInteger(timeout) || timeout < 1 || !Number.isSafeInteger(maximumOutput) || maximumOutput < 1) {
    throw new Error('Invalid SQLite compiler process bounds');
  }
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, {
      ...spawnOptions, detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
    });
    let failure, bytes = 0, finished = false;
    const killTree = () => {
      if (!child.pid) {
        return;
      }
      if (process.platform === 'win32') {
        spawnSync(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32/taskkill.exe'),
          ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore', timeout: 5000, windowsHide: true });
      } else {
        try {
          process.kill(-child.pid, 'SIGKILL');
        } catch (error) {
          if (error.code !== 'ESRCH') {
            child.kill('SIGKILL');
          }
        }
      }
    };
    const stop = error => {
      if (!failure) {
        failure = error;
      }
      killTree();
    };
    const exit = () => killTree();
    const interrupt = () => stop(new Error('SQLite source compilation interrupted'));
    process.once('exit', exit);
    process.once('SIGINT', interrupt);
    process.once('SIGTERM', interrupt);
    const timer = setTimeout(() => stop(new Error('SQLite source compilation exceeded its time limit')), timeout);
    const relay = destination => chunk => {
      bytes += chunk.length;
      if (bytes > maximumOutput) {
        stop(new Error('SQLite compiler output exceeded its byte limit'));
      } else {
        destination.write(chunk);
      }
    };
    child.stdout.on('data', relay(output));
    child.stderr.on('data', relay(errorOutput));
    child.once('error', error => stop(new Error('SQLite compiler could not start', { cause: error })));
    child.once('exit', killTree);
    child.once('close', (code, signal) => {
      if (finished) {
        return;
      }
      finished = true;
      clearTimeout(timer);
      // A child that exits early must not leave a compiler descendant writing
      // after the caller commits or removes its temporary build directory.
      killTree();
      process.removeListener('exit', exit);
      process.removeListener('SIGINT', interrupt);
      process.removeListener('SIGTERM', interrupt);
      if (failure) {
        reject(failure);
      } else if (code !== 0) {
        reject(new Error(`SQLite compiler failed (${ signal || code })`));
      } else {
        resolve();
      }
    });
  });
}
