import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import type { DraftSections } from '../common/report-sections';
import { SECTION_LABELS } from '../common/report-sections';
import type { OutputSection } from '../types/inferenceApi';

@Component({
  standalone: true,
  selector: 'app-report-draft-editor',
  imports: [FormsModule],
  template: `
    <div class="draft-editor" [class.read-only]="readOnly">
      @for (section of sections; track section) {
        <div class="declared-section">
          <label [for]="'report-' + section + '-' + instanceId">{{ sectionLabel(section) }}</label>
          @if (readOnly) {
            <div class="report-section-value">{{ value(section) || 'No report text.' }}</div>
          } @else {
            <textarea
              [id]="'report-' + section + '-' + instanceId"
              [ngModel]="value(section)"
              (ngModelChange)="draftChange.emit({ section, value: $event })"
              [placeholder]="sectionLabel(section) + ' will appear here.'"
            ></textarea>
          }
        </div>
      }
    </div>
  `,
  styleUrl: '../styles/ReportDraftEditor.css',
})
export class ReportDraftEditorComponent {
  @Input() sections: OutputSection[] = [];
  @Input() drafts: DraftSections = {};
  @Input() readOnly = false;
  @Output() readonly draftChange = new EventEmitter<{ section: OutputSection; value: string }>();

  readonly instanceId = Math.random().toString(36).slice(2, 8);

  value(section: OutputSection): string {
    return this.drafts[section] ?? '';
  }

  sectionLabel(section: OutputSection): string {
    return SECTION_LABELS[section];
  }
}
