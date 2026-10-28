import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { TokenizerApiService } from '../services/tokenizer-api.service';
import { TokenizerSelectorComponent } from './tokenizer-selector.component';

@Component({
  standalone: true,
  imports: [TokenizerSelectorComponent],
  template: '<app-tokenizer-selector [value]="value" (valueChange)="value = $event" (compatibleChange)="compatible = $event" />',
})
class HostComponent {
  value = 'distilbert-base-uncased';
  compatible: boolean | null = null;
}

const CATALOGUE = {
  tokenizers: [
    { tokenizer_id: 'distilbert-base-uncased', label: 'DistilBERT base uncased', group: 'BERT family', kind: 'wordpiece', supports_training: true, cached: true, note: null },
    { tokenizer_id: 'dmis-lab/biobert-v1.1', label: 'BioBERT v1.1', group: 'Biomedical and clinical', kind: 'wordpiece', supports_training: true, cached: false, note: null },
    { tokenizer_id: 't5-small', label: 'T5 small', group: 'T5 family', kind: 'sentencepiece', supports_training: false, cached: false, note: 'No start token' },
  ],
};

function checked(overrides: Record<string, unknown> = {}) {
  return { tokenizer: 'x', available: true, compatible: true, cached: true, message: 'ok', vocabulary_size: 30522, start_token: '[CLS]', end_token: '[SEP]', pad_token: '[PAD]', start_token_id: 101, end_token_id: 102, pad_token_id: 0, ...overrides };
}

async function render(check = vi.fn().mockResolvedValue({ result: checked(), error: null })) {
  await TestBed.configureTestingModule({
    imports: [HostComponent],
    providers: [{ provide: TokenizerApiService, useValue: { list: () => Promise.resolve({ result: CATALOGUE, error: null }), check } }],
  }).compileComponents();
  const fixture = TestBed.createComponent(HostComponent);
  fixture.detectChanges();
  await fixture.whenStable();
  fixture.detectChanges();
  return { fixture, check };
}

describe('TokenizerSelectorComponent', () => {
  it('groups tokenizers by family, disables unsupported ones and filters by search', async () => {
    const { fixture } = await render();
    const element = fixture.nativeElement as HTMLElement;

    expect([...element.querySelectorAll('optgroup')].map((group) => group.label)).toEqual(['BERT family', 'Biomedical and clinical', 'T5 family']);
    const t5 = [...element.querySelectorAll('option')].find((option) => option.value === 't5-small') as HTMLOptionElement;
    expect(t5.disabled).toBe(true);
    expect(t5.textContent).toContain('not supported for training');

    const search = element.querySelector('#dataset-tokenizer-search') as HTMLInputElement;
    search.value = 'bio';
    search.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect([...element.querySelectorAll('optgroup')].map((group) => group.label)).toEqual(['BERT family', 'Biomedical and clinical']);
  });

  it('verifies the selected tokenizer and reports compatibility to the parent', async () => {
    const { fixture, check } = await render();
    await new Promise((resolve) => setTimeout(resolve, 10));
    fixture.detectChanges();

    expect(check).toHaveBeenCalledWith('distilbert-base-uncased');
    expect(fixture.componentInstance.compatible).toBe(true);
    expect(fixture.nativeElement.querySelector('.tokenizer-feedback').textContent).toMatch(/Vocabulary 30[.,]522/);
  });

  it('rejects malformed custom ids locally and shows backend incompatibility messages', async () => {
    const check = vi.fn().mockResolvedValue({ result: checked({ compatible: false, message: 'Tokenizer defines no start (CLS/BOS) token' }), error: null });
    const { fixture } = await render(check);
    const select = fixture.nativeElement.querySelector('#dataset-tokenizer') as HTMLSelectElement;
    select.value = '__custom__';
    select.dispatchEvent(new Event('change'));
    fixture.detectChanges();

    const custom = fixture.nativeElement.querySelector('#dataset-tokenizer-custom') as HTMLInputElement;
    custom.value = '../outside';
    custom.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.tokenizer-feedback').textContent).toContain('Use a Hugging Face id');
    expect(fixture.componentInstance.compatible).toBe(false);
    expect(check).not.toHaveBeenCalledWith('../outside');

    custom.value = 'org/no-start';
    custom.dispatchEvent(new Event('input'));
    await new Promise((resolve) => setTimeout(resolve, 450));
    fixture.detectChanges();
    expect(check).toHaveBeenCalledWith('org/no-start');
    expect(fixture.nativeElement.querySelector('.tokenizer-feedback').textContent).toContain('no start');
    expect(fixture.componentInstance.compatible).toBe(false);
  });
});
