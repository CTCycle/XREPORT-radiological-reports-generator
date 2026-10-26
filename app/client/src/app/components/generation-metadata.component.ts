import { JsonPipe } from '@angular/common';
import { Component, computed, input } from '@angular/core';
import { buildMetadataGroups } from '../common/generation-metadata';

@Component({
  selector: 'app-generation-metadata',
  imports: [JsonPipe],
  template: `
    @for (group of groups(); track $index) {
      <section class="metadata-group">
        <h3>{{ group.title }}</h3>
        <dl>
          @for (field of group.fields; track $index) {
            <dt>{{ field.label }}</dt>
            <dd>{{ field.value }}</dd>
          }
        </dl>
      </section>
    } @empty {
      <p class="metadata-empty">No generation metadata was stored for this report.</p>
    }
    <details class="metadata-raw">
      <summary>Raw JSON</summary>
      <pre>{{ config() | json }}</pre>
    </details>
  `,
  styleUrl: '../styles/GenerationMetadata.css',
})
export class GenerationMetadataComponent {
  readonly config = input<Record<string, unknown> | null | undefined>(null);
  readonly groups = computed(() => buildMetadataGroups(this.config()));
}
