import { buildMetadataGroups, formatMetadataValue } from './generation-metadata';

describe('generation metadata', () => {
  it('formats dimensions, tensors and booleans for display', () => {
    expect(formatMetadataValue({ width: 512, height: 512 })).toBe('512 × 512 px');
    expect(formatMetadataValue([1, 1, 3, 384, 384])).toBe('1 × 1 × 3 × 384 × 384');
    expect(formatMetadataValue(false)).toBe('No');
    expect(formatMetadataValue(null)).toBe('—');
  });

  it('groups run, per-image, error and other fields', () => {
    const groups = buildMetadataGroups({
      profile: 'concise',
      inference_metadata: [{ filename: 'a.jpg', cuda_used: true }],
      provenance: { report_scope: 'study', image_errors: { 'b.jpg': 'decode failed' } },
      extra_field: 3,
    });

    expect(groups.map((group) => group.title)).toEqual(['Run', 'Image 1 · a.jpg', 'Image errors', 'Other']);
    expect(groups[0].fields).toEqual([{ label: 'Profile', value: 'concise' }, { label: 'Report scope', value: 'study' }]);
    expect(groups[1].fields).toEqual([{ label: 'CUDA used', value: 'Yes' }]);
  });

  it('tolerates malformed configs', () => {
    expect(buildMetadataGroups({ provenance: 'test' })).toEqual([]);
    expect(buildMetadataGroups(null)).toEqual([]);
  });
});
