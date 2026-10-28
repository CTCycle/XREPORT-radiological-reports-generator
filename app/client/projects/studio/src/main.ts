import { bootstrapApplication } from '@angular/platform-browser';
import { App } from '@xreport/shared/app/app';
import { markStartupPhase } from '@xreport/shared/app/services/startup-timing';
import { studioConfig } from './studio.config';

markStartupPhase('frontend_bootstrap_started');
bootstrapApplication(App, studioConfig)
  .then(() => markStartupPhase('frontend_bootstrap_completed'))
  .catch((err) => console.error(err));
