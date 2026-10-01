import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import type { DraftSections } from '../common/report-sections';
import type { OutputSection } from '../types/inferenceApi';
import { ReportDraftEditorComponent } from './report-draft-editor.component';

@Component({
  standalone: true,
  imports: [ReportDraftEditorComponent],
  template: `
    <app-report-draft-editor
      [sections]="sections"
      [drafts]="drafts"
      [readOnly]="readOnly"
      (draftChange)="lastChange = $event"
    />
  `,
})
class EditorHostComponent {
  sections: OutputSection[] = ['findings', 'impression'];
  drafts: DraftSections = { findings: 'Clear lungs.', impression: 'No acute disease.' };
  readOnly = false;
  lastChange: { section: OutputSection; value: string } | null = null;
}

describe('ReportDraftEditorComponent', () => {
  it('renders declared sections and emits controlled text changes', async () => {
    const fixture = TestBed.createComponent(EditorHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    const element = fixture.nativeElement as HTMLElement;

    expect(element.querySelectorAll('textarea')).toHaveLength(2);
    expect(element.textContent).toContain('Findings');
    expect((element.querySelector('textarea') as HTMLTextAreaElement).value).toBe('Clear lungs.');

    const findings = element.querySelector('textarea') as HTMLTextAreaElement;
    findings.value = 'Clear lungs after review.';
    findings.dispatchEvent(new Event('input'));
    fixture.detectChanges();

    expect(fixture.componentInstance.lastChange).toEqual({
      section: 'findings',
      value: 'Clear lungs after review.',
    });
  });

  it('renders read-only report text without editable controls', () => {
    const fixture = TestBed.createComponent(EditorHostComponent);
    fixture.componentInstance.readOnly = true;
    fixture.detectChanges();
    const element = fixture.nativeElement as HTMLElement;

    expect(element.querySelectorAll('textarea')).toHaveLength(0);
    expect(element.querySelectorAll('.report-section-value')).toHaveLength(2);
    expect(element.textContent).toContain('No acute disease.');
  });
});
