import { Component, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucidePause, lucidePlay } from '@ng-icons/lucide';

@Component({
  standalone: true,
  selector: 'app-tutorial-media',
  imports: [NgIcon],
  providers: [provideIcons({ lucidePause, lucidePlay })],
  template: `
    <div class="tutorial-media" [class.paused]="!playing()" aria-label="Inference workflow demonstration">
      <div class="tutorial-media-flow" aria-hidden="true">
        @for (step of steps; track step; let index = $index) {
          <div class="tutorial-media-step">
            <span class="tutorial-media-node" [style.animation-delay]="(index * 1.4) + 's'">{{ index + 1 }}</span>
            <span>{{ step }}</span>
          </div>
        }
      </div>
      <p class="sr-only">Choose a model, add study images, generate a draft, and review the editable result.</p>
      <div class="tutorial-media-controls">
        <button type="button" [attr.aria-pressed]="!playing()" (click)="toggle()">
          <ng-icon [name]="playing() ? 'lucidePause' : 'lucidePlay'" size="13" aria-hidden="true" />
          {{ playing() ? 'Pause demo' : 'Play demo' }}
        </button>
      </div>
    </div>
  `,
  styleUrl: '../styles/Guidance.css',
})
export class TutorialMediaComponent {
  readonly steps = ['Choose', 'Add', 'Generate', 'Review'];
  readonly playing = signal(true);

  toggle(): void {
    this.playing.update((playing) => !playing);
  }
}
