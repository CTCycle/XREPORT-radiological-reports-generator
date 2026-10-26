import { diffOriginal } from './text-diff';

describe('diffOriginal', () => {
  it('reports no changes for identical text', () => {
    expect(diffOriginal('Heart size is normal.', 'Heart size is normal.')).toEqual([
      { text: 'Heart size is normal.', changed: false },
    ]);
  });

  it('flags original words that were replaced or removed', () => {
    expect(diffOriginal('No acute disease seen.', 'No chronic disease.')).toEqual([
      { text: 'No ', changed: false },
      { text: 'acute', changed: true },
      { text: ' disease ', changed: false },
      { text: 'seen', changed: true },
      { text: '.', changed: false },
    ]);
  });

  it('ignores pure insertions and merges adjacent changed words into one run', () => {
    expect(diffOriginal('Lungs are clear', 'Lungs are clear today')).toEqual([
      { text: 'Lungs are clear', changed: false },
    ]);
    expect(diffOriginal('a b c d', 'a d')).toEqual([
      { text: 'a ', changed: false },
      { text: 'b c', changed: true },
      { text: ' d', changed: false },
    ]);
  });
});
