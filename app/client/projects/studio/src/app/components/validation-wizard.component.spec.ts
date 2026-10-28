import { TestBed } from '@angular/core/testing';
import { TokenizerApiService } from '../services/tokenizer-api.service';
import type { ValidationWizardConfirmPayload } from '../types/validationWizard';
import { ValidationWizardComponent } from './validation-wizard.component';

const CHECKED = { tokenizer: 'distilbert-base-uncased', available: true, compatible: true, cached: true, message: 'ok', vocabulary_size: 30522, start_token: '[CLS]', end_token: '[SEP]', pad_token: '[PAD]', start_token_id: 101, end_token_id: 102, pad_token_id: 0 };

describe('ValidationWizardComponent', () => {
  it('requests token counts only for text statistics with a verified tokenizer', async () => {
    await TestBed.configureTestingModule({
      imports: [ValidationWizardComponent],
      providers: [{ provide: TokenizerApiService, useValue: { list: () => Promise.resolve({ result: { tokenizers: [] }, error: null }), check: () => Promise.resolve({ result: CHECKED, error: null }) } }],
    }).compileComponents();
    const fixture = TestBed.createComponent(ValidationWizardComponent);
    const component = fixture.componentInstance;
    const payloads: ValidationWizardConfirmPayload[] = [];
    component.confirmed.subscribe((payload) => payloads.push(payload));
    component.open = true;
    component.ngOnChanges({ open: { currentValue: true, previousValue: false, firstChange: true, isFirstChange: () => true } });
    fixture.detectChanges();
    const element = fixture.nativeElement as HTMLElement;

    expect(element.querySelector('#validation-wizard-count-tokens')).toBeNull();
    component.toggleMetric('text_statistics');
    fixture.detectChanges();
    (element.querySelector('#validation-wizard-count-tokens') as HTMLButtonElement).click();
    fixture.detectChanges();
    await new Promise((resolve) => setTimeout(resolve, 500));
    fixture.detectChanges();
    expect(element.querySelector('#validation-wizard-tokenizer')).not.toBeNull();

    component.confirm();
    expect(payloads[0].tokenizer).toBe('distilbert-base-uncased');
    component.toggleMetric('text_statistics');
    component.toggleMetric('image_statistics');
    component.confirm();
    expect(payloads[1].tokenizer).toBeNull();
  });
});
