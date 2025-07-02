/**
 * Memory configuration for cognitive patterns
 * Separated to avoid circular dependencies
 */

// Memory configuration for optimized patterns
const PATTERN_MEMORY_CONFIG = {
  convergent: {
    baseMemory: 250, // Reduced from 291 MB
    poolSharing: 0.8, // 80% shared memory
    lazyLoading: true,
  },
  divergent: {
    baseMemory: 280, // Reduced from 473 MB
    poolSharing: 0.7, // 70% shared memory
    lazyLoading: true,
  },
  lateral: {
    baseMemory: 300, // Reduced from 557 MB
    poolSharing: 0.65, // 65% shared memory
    lazyLoading: true,
  },
  systems: {
    baseMemory: 270,
    poolSharing: 0.75,
    lazyLoading: true,
  },
  critical: {
    baseMemory: 260,
    poolSharing: 0.75,
    lazyLoading: true,
  },
  abstract: {
    baseMemory: 265,
    poolSharing: 0.75,
    lazyLoading: true,
  },
};

module.exports = {
  PATTERN_MEMORY_CONFIG,
};