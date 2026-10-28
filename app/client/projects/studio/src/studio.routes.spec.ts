import { describe, expect, it } from 'vitest';
import { studioRoutes } from './studio.routes';

describe('XREPORT Studio route set', () => {
  it('serves dataset, model and training pages only', () => {
    const paths = (studioRoutes[0].children ?? []).map((route) => route.path);

    expect(paths).toEqual(['', 'dataset', 'dataset/validate/:datasetName', 'model', 'training', 'settings', '**']);
    expect(studioRoutes[0].data?.['application']).toBe('studio');
  });
});
