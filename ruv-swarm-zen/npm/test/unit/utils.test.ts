import {
  generateId,
  getDefaultCognitiveProfile,
  calculateCognitiveDiversity,
  recommendTopology,
  priorityToNumber,
  validateSwarmOptions,
} from '../src/utils';

describe('Utils', () => {
  describe('generateId', () => {
    it('should generate unique IDs', () => {
      const id1 = generateId();
      const id2 = generateId();
      expect(id1).not.toBe(id2);
    });

    it('should include prefix when provided', () => {
      const id = generateId('test');
      expect(id).toMatch(/^test_/);
    });
  });

  describe('getDefaultCognitiveProfile', () => {
    it('should return correct profile for researcher', () => {
      const profile = getDefaultCognitiveProfile('researcher');
      expect(profile.analytical).toBeGreaterThan(0.8);
      expect(profile.systematic).toBeGreaterThan(0.7);
    });

    it('should return balanced profile for custom type', () => {
      const profile = getDefaultCognitiveProfile('custom');
      expect(profile.analytical).toBe(0.5);
      expect(profile.creative).toBe(0.5);
    });
  });

  describe('calculateCognitiveDiversity', () => {
    it('should return 0 for identical profiles', () => {
      const profile = getDefaultCognitiveProfile('researcher');
      const diversity = calculateCognitiveDiversity(profile, profile);
      expect(diversity).toBe(0);
    });

    it('should return positive value for different profiles', () => {
      const profile1 = getDefaultCognitiveProfile('researcher');
      const profile2 = getDefaultCognitiveProfile('coder');
      const diversity = calculateCognitiveDiversity(profile1, profile2);
      expect(diversity).toBeGreaterThan(0);
    });
  });

  describe('recommendTopology', () => {
    it('should recommend mesh for small agent count', () => {
      const topology = recommendTopology(3, 'medium', 'moderate');
      expect(topology).toBe('mesh');
    });

    it('should recommend hierarchical for extensive coordination', () => {
      const topology = recommendTopology(10, 'high', 'extensive');
      expect(topology).toBe('hierarchical');
    });

    it('should recommend distributed for minimal coordination', () => {
      const topology = recommendTopology(8, 'low', 'minimal');
      expect(topology).toBe('distributed');
    });
  });

  describe('priorityToNumber', () => {
    it('should convert priorities correctly', () => {
      expect(priorityToNumber('low')).toBe(1);
      expect(priorityToNumber('medium')).toBe(2);
      expect(priorityToNumber('high')).toBe(3);
      expect(priorityToNumber('critical')).toBe(4);
    });
  });

  describe('validateSwarmOptions', () => {
    it('should return empty array for valid options', () => {
      const errors = validateSwarmOptions({
        maxAgents: 10,
        connectionDensity: 0.5,
        topology: 'mesh',
      });
      expect(errors).toEqual([]);
    });

    it('should return errors for invalid maxAgents', () => {
      const errors = validateSwarmOptions({ maxAgents: -1 });
      expect(errors).toContain('maxAgents must be a positive number');
    });

    it('should return errors for invalid connectionDensity', () => {
      const errors = validateSwarmOptions({ connectionDensity: 2 });
      expect(errors).toContain('connectionDensity must be a number between 0 and 1');
    });

    it('should return errors for invalid topology', () => {
      const errors = validateSwarmOptions({ topology: 'invalid' });
      expect(errors[0]).toMatch(/topology must be one of/);
    });
  });
});