import type { OutputSection } from '../types/inferenceApi';

export type DraftSections = Partial<Record<OutputSection, string>>;

export const SECTION_LABELS: Record<OutputSection, string> = {
  raw_report: 'Raw report',
  findings: 'Findings',
  impression: 'Impression',
};

export function isOutputSection(value: string): value is OutputSection {
  return value === 'raw_report' || value === 'findings' || value === 'impression';
}

export function parseReportDraft(report: string, sections: OutputSection[]): DraftSections {
  if (sections.includes('raw_report')) return { raw_report: report };

  const draft: DraftSections = {};
  const normalized = report.trim();
  const findings = normalized.match(
    /(?:^|\n)\s*(?:#{1,3}\s*)?findings\s*:?\s*([\s\S]*?)(?=\n\s*(?:#{1,3}\s*)?impression\s*:?|$)/i,
  )?.[1]?.trim();
  const impression = normalized.match(
    /(?:^|\n)\s*(?:#{1,3}\s*)?impression\s*:?\s*([\s\S]*)$/i,
  )?.[1]?.trim();

  if (sections.includes('findings')) {
    draft.findings = findings ?? (sections.length === 1 ? normalized : '');
  }
  if (sections.includes('impression')) {
    draft.impression = impression ?? (sections.length === 1 ? normalized : '');
  }
  return draft;
}

export function formatReportDraft(
  sections: OutputSection[],
  drafts: DraftSections,
  fallback = '',
): string {
  if (!sections.length) return fallback;
  return sections
    .map((section) => `${SECTION_LABELS[section]}\n${(drafts[section] ?? '').trim()}`)
    .join('\n\n')
    .trim();
}
