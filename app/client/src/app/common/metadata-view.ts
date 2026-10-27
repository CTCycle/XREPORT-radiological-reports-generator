import { formatMetadataLabel, formatMetadataValue } from './metadata';

export interface MetadataViewRow { label: string; value: string; mono?: boolean; }
export interface MetadataViewGroup { title: string; rows: MetadataViewRow[]; }

interface FieldSpec { label: string; format?: (value: unknown) => string; mono?: boolean; }

const percent = (value: unknown): string => typeof value === 'number' ? `${Number((value * 100).toFixed(2))}%` : formatMetadataValue(value);
const tokens = (value: unknown): string => typeof value === 'number' ? `${value.toLocaleString('en-US')} tokens` : formatMetadataValue(value);
const count = (value: unknown): string => typeof value === 'number' ? value.toLocaleString('en-US') : formatMetadataValue(value);

const DATASET_GROUPS: { title: string; fields: Record<string, FieldSpec> }[] = [
  { title: 'Source', fields: { source_dataset: { label: 'Source dataset' }, dataset_name: { label: 'Dataset name' }, date: { label: 'Processed on' } } },
  { title: 'Sampling and split', fields: { sample_size: { label: 'Sample fraction', format: percent }, seed: { label: 'Sampling seed', format: count }, validation_size: { label: 'Validation fraction', format: percent }, split_seed: { label: 'Split seed', format: count } } },
  { title: 'Tokenization', fields: { tokenizer: { label: 'Tokenizer', mono: true }, vocabulary_size: { label: 'Vocabulary size', format: count }, max_report_size: { label: 'Maximum report length', format: tokens } } },
  { title: 'Integrity', fields: { hashcode: { label: 'Hash code', mono: true } } },
];

const flatten = (value: unknown, prefix: string, rows: MetadataViewRow[], depth = 0): void => {
  if (value && typeof value === 'object' && !Array.isArray(value) && depth < 3) {
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) flatten(child, prefix ? `${prefix} · ${formatMetadataLabel(key)}` : formatMetadataLabel(key), rows, depth + 1);
    return;
  }
  if (Array.isArray(value) && value.every((item) => item === null || typeof item !== 'object')) rows.push({ label: prefix, value: value.length ? value.map(String).join(', ') : 'None' });
  else rows.push({ label: prefix, value: formatMetadataValue(value), mono: typeof value === 'object' });
};

/** Builds labelled groups from dataset processing metadata; unknown keys fall into "Other" so no value is hidden. */
export function buildDatasetMetadataGroups(metadata: Record<string, unknown>): MetadataViewGroup[] {
  const used = new Set<string>();
  const groups: MetadataViewGroup[] = DATASET_GROUPS.map((group) => {
    const rows = Object.entries(group.fields).filter(([key]) => key in metadata).map(([key, spec]) => { used.add(key); return { label: spec.label, value: spec.format ? spec.format(metadata[key]) : formatMetadataValue(metadata[key]), mono: spec.mono }; });
    return { title: group.title, rows };
  });
  const other: MetadataViewRow[] = [];
  for (const [key, value] of Object.entries(metadata)) if (!used.has(key)) flatten(value, formatMetadataLabel(key), other);
  if (other.length) groups.push({ title: 'Other', rows: other });
  return groups.filter((group) => group.rows.length);
}

/** Builds one group per top-level section of the checkpoint metadata response. */
export function buildCheckpointMetadataGroups(response: Record<string, unknown>): MetadataViewGroup[] {
  const groups: MetadataViewGroup[] = [];
  const general: MetadataViewRow[] = [];
  for (const [key, value] of Object.entries(response)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      const rows: MetadataViewRow[] = [];
      flatten(value, '', rows, 1);
      if (rows.length) groups.push({ title: formatMetadataLabel(key), rows: rows.map((row) => ({ ...row, label: row.label || formatMetadataLabel(key) })) });
    } else flatten(value, formatMetadataLabel(key), general);
  }
  return [...(general.length ? [{ title: 'General', rows: general }] : []), ...groups];
}
