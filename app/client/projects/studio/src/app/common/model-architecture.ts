import type { ArchitectureFile, ModelArchitecture } from '../types/modelArchitectureApi';

export const ARCHITECTURE_FILE_FORMAT = 'xreport-model-architecture';

/** Compact parameter count, e.g. 85.8M or 412K. */
export function formatParameters(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(0)}K`;
  return String(count);
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

export function buildArchitectureFile(name: string, description: string, architecture: ModelArchitecture): ArchitectureFile {
  return { format: ARCHITECTURE_FILE_FORMAT, version: 1, name, description, architecture };
}

/**
 * Parse an exported preset file. Field values are validated by the server when
 * the preset is saved; this only checks the envelope.
 */
export function parseArchitectureFile(text: string): { name: string; description: string; architecture: Record<string, unknown> } {
  let parsed: unknown;
  try { parsed = JSON.parse(text); } catch { throw new Error('The file is not valid JSON.'); }
  const file = parsed as Partial<ArchitectureFile> | null;
  if (!file || typeof file !== 'object' || file.format !== ARCHITECTURE_FILE_FORMAT) throw new Error('Not an XREPORT model architecture file.');
  if (file.version !== 1) throw new Error(`Unsupported architecture file version: ${String(file.version)}.`);
  if (typeof file.name !== 'string' || !file.name.trim()) throw new Error('The architecture file has no preset name.');
  if (!file.architecture || typeof file.architecture !== 'object' || Array.isArray(file.architecture)) throw new Error('The architecture file has no architecture block.');
  return { name: file.name.trim(), description: typeof file.description === 'string' ? file.description : '', architecture: file.architecture as Record<string, unknown> };
}
