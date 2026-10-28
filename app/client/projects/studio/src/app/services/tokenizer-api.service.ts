import { Injectable, inject } from '@angular/core';
import type { TokenizerCatalogueResponse, TokenizerCheckResponse } from '@xreport/shared/app/types/generated/studio/types.gen';
import { ApiRequestService } from '@xreport/shared/app/services/api-request.service';

@Injectable({ providedIn: 'root' })
export class TokenizerApiService {
  private readonly request = inject(ApiRequestService);

  list() { return this.request.request<TokenizerCatalogueResponse>('GET', '/api/tokenizers'); }
  check(tokenizer: string) { return this.request.request<TokenizerCheckResponse>('POST', '/api/tokenizers/check', { tokenizer }); }
}
