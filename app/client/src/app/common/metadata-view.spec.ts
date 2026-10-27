import { describe, expect, it } from 'vitest';
import { buildCheckpointMetadataGroups, buildDatasetMetadataGroups } from './metadata-view';

describe('metadata view', () => {
  it('groups dataset processing metadata with readable labels and keeps unknown keys', () => {
    const groups = buildDatasetMetadataGroups({ source_dataset: 'reports', hashcode: 'abc123', sample_size: 1, validation_size: 0.2, vocabulary_size: 30522, tokenizer: 'distilbert-base-uncased', custom_flag: true });
    const rows = Object.fromEntries(groups.flatMap((group) => group.rows.map((row) => [row.label, row.value])));
    expect(groups.map((group) => group.title)).toEqual(['Source', 'Sampling and split', 'Tokenization', 'Integrity', 'Other']);
    expect(rows).toMatchObject({ 'Source dataset': 'reports', 'Sample fraction': '100%', 'Validation fraction': '20%', 'Vocabulary size': '30,522', 'Hash code': 'abc123', 'Custom Flag': 'true' });
  });

  it('turns each checkpoint section into a group with flattened nested values', () => {
    const groups = buildCheckpointMetadataGroups({ checkpoint: 'ckpt', configuration: { epochs: 5, optimizer: { name: 'adam' } }, session: {} });
    expect(groups.map((group) => group.title)).toEqual(['General', 'Configuration']);
    expect(groups[1].rows).toEqual([{ label: 'Epochs', value: '5', mono: false }, { label: 'Optimizer · Name', value: 'adam', mono: false }]);
  });
});
