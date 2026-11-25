import { describe, expect, it } from 'vitest';

import {
  formatReportDraft,
  formatReportExport,
  parseReportDraft,
  RESEARCH_USE_NOTICE,
  SECTION_LABELS,
} from './report-sections';

describe('report section helpers', () => {
  it('keeps raw-report output intact', () => {
    const report = 'Free-form model output\nwith multiple lines.';

    expect(parseReportDraft(report, ['raw_report'])).toEqual({ raw_report: report });
    expect(formatReportDraft(['raw_report'], { raw_report: report })).toBe(
      `Raw report\n${report}`,
    );
  });

  it('parses declared findings and impression sections', () => {
    const drafts = parseReportDraft(
      'Findings:\nNo focal opacity.\n\nImpression:\nNo acute cardiopulmonary abnormality.',
      ['findings', 'impression'],
    );

    expect(drafts).toEqual({
      findings: 'No focal opacity.',
      impression: 'No acute cardiopulmonary abnormality.',
    });
  });

  it('formats editable sections with stable labels and spacing', () => {
    expect(SECTION_LABELS.findings).toBe('Findings');
    expect(
      formatReportDraft(
        ['findings', 'impression'],
        { findings: 'No focal opacity.', impression: 'No acute abnormality.' },
      ),
    ).toBe('Findings\nNo focal opacity.\n\nImpression\nNo acute abnormality.');
  });

  it('prefixes exported text with research status and study provenance', () => {
    const text = formatReportExport('Findings\nClear lungs.', {
      model: 'CheXOne',
      revision: 'abc123',
      sources: ['pa.png', 'lateral.png'],
      studyReport: true,
      exportedAt: '2026-11-25T10:00:00.000Z',
    });

    expect(text.split('\n')).toEqual([
      RESEARCH_USE_NOTICE,
      'Model: CheXOne (revision abc123)',
      'Source (2 images read as one study): pa.png, lateral.png',
      'Exported: 2026-11-25T10:00:00.000Z',
      '',
      'Findings',
      'Clear lungs.',
    ]);
  });
});
