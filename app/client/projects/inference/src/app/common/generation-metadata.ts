export interface MetadataField {
  readonly label: string;
  readonly value: string;
}

export interface MetadataGroup {
  readonly title: string;
  readonly fields: readonly MetadataField[];
}

const LABELS: Record<string, string> = {
  original_dimensions: 'Original size',
  processed_tensor_dimensions: 'Model input shape',
  cuda_available: 'CUDA available',
  cuda_used: 'CUDA used',
  model_dtype: 'Model dtype',
  report_scope: 'Report scope',
};
const RUN_EXCLUDED = new Set(['input_images', 'image_errors']);
const MISSING = '—';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function metadataLabel(key: string): string {
  const label = LABELS[key] ?? key.replace(/_/g, ' ');
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export function formatMetadataValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return MISSING;
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  if (Array.isArray(value)) {
    if (!value.length) return MISSING;
    return value.every((item) => typeof item === 'number') ? value.join(' × ') : value.map(formatMetadataValue).join(', ');
  }
  if (isRecord(value) && typeof value['width'] === 'number' && typeof value['height'] === 'number') {
    return `${value['width']} × ${value['height']} px`;
  }
  return JSON.stringify(value);
}

function fieldsOf(record: Record<string, unknown>, excluded: ReadonlySet<string> = new Set()): MetadataField[] {
  return Object.entries(record)
    .filter(([key]) => !excluded.has(key))
    .map(([key, value]) => ({ label: metadataLabel(key), value: formatMetadataValue(value) }));
}

/** Turns the stored generation config into labelled groups; the raw JSON stays untouched. */
export function buildMetadataGroups(config: Record<string, unknown> | null | undefined): MetadataGroup[] {
  if (!config) return [];
  const groups: MetadataGroup[] = [];
  const provenance = isRecord(config['provenance']) ? config['provenance'] : {};
  const run = [
    ...fieldsOf({ profile: config['profile'] }),
    ...fieldsOf(provenance, RUN_EXCLUDED),
  ].filter((field) => field.value !== MISSING);
  if (run.length) groups.push({ title: 'Run', fields: run });

  const images = Array.isArray(config['inference_metadata']) ? config['inference_metadata'] : [];
  images.filter(isRecord).forEach((entry, index) => {
    const name = typeof entry['filename'] === 'string' ? entry['filename'] : '';
    groups.push({ title: `Image ${index + 1}${name ? ` · ${name}` : ''}`, fields: fieldsOf(entry, new Set(['filename'])) });
  });

  const errors = isRecord(provenance['image_errors']) ? provenance['image_errors'] : {};
  if (Object.keys(errors).length) {
    groups.push({ title: 'Image errors', fields: Object.entries(errors).map(([name, message]) => ({ label: name, value: String(message) })) });
  }

  const known = new Set(['profile', 'display_sections', 'provenance', 'inference_metadata']);
  const other = fieldsOf(config, known);
  if (other.length) groups.push({ title: 'Other', fields: other });
  return groups;
}
