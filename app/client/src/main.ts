import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { App } from './app/app';
import { markStartupPhase } from './app/services/startup-timing';

markStartupPhase('frontend_bootstrap_started');
bootstrapApplication(App, appConfig)
  .then(() => markStartupPhase('frontend_bootstrap_completed'))
  .catch((err) => console.error(err));
