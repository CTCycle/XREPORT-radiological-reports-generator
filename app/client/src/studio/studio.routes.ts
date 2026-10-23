import { Routes } from '@angular/router';

export const studioRoutes: Routes = [
  {
    path: '',
    data: { application: 'studio' },
    loadComponent: () => import('../app/components/main-layout.component').then((m) => m.MainLayoutComponent),
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'dataset' },
      { path: 'dataset', loadComponent: () => import('../app/pages/dataset.page').then((m) => m.DatasetPage) },
      { path: 'dataset/validate/:datasetName', loadComponent: () => import('../app/pages/dataset-validation.page').then((m) => m.DatasetValidationPage) },
      { path: 'training', loadComponent: () => import('../app/pages/training.page').then((m) => m.TrainingPage) },
      { path: 'settings', loadComponent: () => import('../app/pages/settings.page').then((m) => m.SettingsPage) },
      { path: '**', redirectTo: 'dataset' },
    ],
  },
];
