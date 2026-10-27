import { describe, expect, it } from 'vitest';
import { stackChips } from '../components/architecture-diagram.component';
import { buildArchitectureFile, parseArchitectureFile } from './model-architecture';
import type { ModelArchitecture } from '../types/modelArchitectureApi';

const architecture = { embedding_dims: 256, num_encoders: 2 } as unknown as ModelArchitecture;

describe('architecture file', () => {
  it('round-trips an exported preset', () => {
    const exported = JSON.stringify(buildArchitectureFile(' Narrow ', 'small', architecture));
    expect(parseArchitectureFile(exported)).toEqual({ name: 'Narrow', description: 'small', architecture });
  });

  it('rejects files that are not architecture exports', () => {
    expect(() => parseArchitectureFile('{')).toThrow('not valid JSON');
    expect(() => parseArchitectureFile(JSON.stringify({ name: 'x', architecture }))).toThrow('Not an XREPORT');
    expect(() => parseArchitectureFile(JSON.stringify({ ...buildArchitectureFile('x', '', architecture), version: 2 }))).toThrow('Unsupported');
    expect(() => parseArchitectureFile(JSON.stringify({ ...buildArchitectureFile('x', '', architecture), architecture: [] }))).toThrow('no architecture block');
  });
});

describe('stack chips', () => {
  it('shows every layer up to six and collapses deeper stacks', () => {
    expect(stackChips(3, 'E')).toEqual(['E1', 'E2', 'E3']);
    expect(stackChips(12, 'D')).toEqual(['D1', 'D2', '…', 'D11', 'D12']);
  });
});
