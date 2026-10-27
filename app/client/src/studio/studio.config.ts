import { ApplicationConfig, inject, provideAppInitializer, provideBrowserGlobalErrorListeners } from '@angular/core';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideRouter } from '@angular/router';

import { DatasetTrainingJourneyComponent } from '../app/components/dataset-training-journey.component';
import { APP_NAME } from '../app/components/startup-screen.component';
import { TIPS_JOURNEY } from '../app/components/tips-and-tricks.component';
import { ThemeService } from '../app/services/theme.service';
import { studioRoutes } from './studio.routes';

export const studioConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideRouter(studioRoutes),
    provideHttpClient(withFetch()),
    { provide: APP_NAME, useValue: 'XREPORT Studio' },
    { provide: TIPS_JOURNEY, useValue: DatasetTrainingJourneyComponent },
    provideAppInitializer(() => {
      inject(ThemeService);
    }),
  ],
};
