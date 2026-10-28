import { Component, EventEmitter, Input, OnInit, Output, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TokenizerApiService } from '../services/tokenizer-api.service';
import type { TokenizerCheckResponse, TokenizerOption } from '@xreport/shared/app/types/generated/studio/types.gen';

const CUSTOM = '__custom__';
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/;
const CHECK_DELAY_MS = 400;

/** Searchable, grouped tokenizer picker that verifies the chosen tokenizer with the backend. */
@Component({
  standalone: true,
  selector: 'app-tokenizer-selector',
  imports: [FormsModule],
  template: `
    <div class="form-group tokenizer-selector">
      <label class="form-label" for="dataset-tokenizer-search">Tokenizer</label>
      <input id="dataset-tokenizer-search" class="form-input" type="search" placeholder="Search by name, family or id" autocomplete="off" [ngModel]="search()" (ngModelChange)="search.set($event)"/>
      <select id="dataset-tokenizer" class="form-select" aria-label="Tokenizer" [ngModel]="selection()" (ngModelChange)="choose($event)">
        @for (group of groups(); track group.name) {
          <optgroup [label]="group.name">
            @for (option of group.options; track option.tokenizer_id) {
              <option [value]="option.tokenizer_id" [disabled]="!option.supports_training">{{ option.label }}{{ option.supports_training ? '' : ' - not supported for training' }}</option>
            }
          </optgroup>
        }
        <option [value]="customValue">Custom Hugging Face tokenizer…</option>
      </select>
      @if (selection() === customValue) {
        <input id="dataset-tokenizer-custom" class="form-input" type="text" placeholder="organisation/tokenizer-name" aria-label="Custom tokenizer id" autocomplete="off" spellcheck="false" [ngModel]="value" (ngModelChange)="setCustom($event)"/>
      }
      <p class="tokenizer-feedback" [class.tokenizer-feedback-error]="feedbackIsError()" role="status" aria-live="polite">{{ feedback() }}</p>
    </div>
  `,
  styles: [`
    .tokenizer-selector { display: flex; flex-direction: column; gap: 0.35rem; }
    .tokenizer-feedback { margin: 0; font-size: 0.75rem; color: var(--text-secondary); overflow-wrap: anywhere; }
    .tokenizer-feedback-error { color: var(--error-color); }
  `],
})
export class TokenizerSelectorComponent implements OnInit {
  private readonly api = inject(TokenizerApiService);
  @Input() set value(next: string) {
    if (next === this.current()) return;
    this.current.set(next ?? '');
    if (this.ready) this.scheduleCheck(0);
  }
  get value() { return this.current(); }
  @Output() readonly valueChange = new EventEmitter<string>();
  /** `true` when the backend verified the tokenizer, `false` when unusable, `null` while unknown. */
  @Output() readonly compatibleChange = new EventEmitter<boolean | null>();

  readonly customValue = CUSTOM;
  private readonly current = signal('');
  private ready = false;
  readonly options = signal<TokenizerOption[]>([]);
  readonly search = signal('');
  readonly customMode = signal(false);
  readonly checking = signal(false);
  readonly result = signal<TokenizerCheckResponse | null>(null);
  readonly localError = signal<string | null>(null);
  readonly catalogueError = signal<string | null>(null);
  private timer: ReturnType<typeof setTimeout> | undefined;
  private sequence = 0;

  readonly groups = computed(() => {
    const term = this.search().trim().toLowerCase();
    const groups = new Map<string, TokenizerOption[]>();
    for (const option of this.options()) {
      const haystack = `${option.label} ${option.tokenizer_id} ${option.group}`.toLowerCase();
      if (term && !haystack.includes(term) && option.tokenizer_id !== this.current()) continue;
      groups.set(option.group, [...(groups.get(option.group) ?? []), option]);
    }
    return [...groups].map(([name, options]) => ({ name, options }));
  });

  readonly selection = computed(() => (this.customMode() || !this.options().some((option) => option.tokenizer_id === this.current()) ? CUSTOM : this.current()));
  readonly feedbackIsError = computed(() => !!this.localError() || !!this.catalogueError() || (!!this.result() && !this.result()!.compatible));
  readonly feedback = computed(() => {
    if (this.localError()) return this.localError()!;
    if (this.checking()) return 'Checking tokenizer…';
    const result = this.result();
    if (!result) return this.catalogueError() ?? '';
    if (!result.compatible) return result.message;
    const tokens = `start ${result.start_token} (${result.start_token_id}), end ${result.end_token} (${result.end_token_id}), pad ${result.pad_token} (${result.pad_token_id})`;
    return `${result.tokenizer}: compatible with the report decoder. Vocabulary ${new Intl.NumberFormat().format(result.vocabulary_size ?? 0)}; ${tokens}.${result.cached ? '' : ' Files will be downloaded on first use.'}`;
  });

  async ngOnInit() {
    const catalogue = await this.api.list();
    if (catalogue.result) this.options.set(catalogue.result.tokenizers);
    else this.catalogueError.set(catalogue.error ?? 'The tokenizer list could not be loaded.');
    this.ready = true;
    this.scheduleCheck(0);
  }

  choose(selected: string) {
    if (selected === CUSTOM) { this.customMode.set(true); this.update(''); return; }
    this.customMode.set(false);
    this.update(selected);
  }

  setCustom(value: string) { this.update(value.trim()); }

  private update(value: string) {
    this.current.set(value);
    this.valueChange.emit(value);
    this.scheduleCheck(CHECK_DELAY_MS);
  }

  private scheduleCheck(delay: number) {
    clearTimeout(this.timer);
    const sequence = ++this.sequence;
    this.result.set(null);
    const tokenizer = this.value;
    if (!tokenizer) { this.localError.set(this.customMode() ? 'Enter a Hugging Face tokenizer id.' : null); this.compatibleChange.emit(null); return; }
    if (!ID_PATTERN.test(tokenizer)) { this.localError.set('Use a Hugging Face id such as "organisation/name" (letters, digits, ".", "_" and "-").'); this.compatibleChange.emit(false); return; }
    this.localError.set(null);
    this.compatibleChange.emit(null);
    this.timer = setTimeout(async () => {
      this.checking.set(true);
      const checked = await this.api.check(tokenizer);
      if (sequence !== this.sequence) return;
      this.checking.set(false);
      this.result.set(checked.result);
      if (!checked.result) this.localError.set(checked.error ?? 'The tokenizer could not be checked.');
      this.compatibleChange.emit(checked.result ? checked.result.available && checked.result.compatible : false);
    }, delay);
  }
}
