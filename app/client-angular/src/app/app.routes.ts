import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./components/main-layout.component').then((m) => m.MainLayoutComponent),
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'inference' },
      { path: 'inference', loadComponent: () => import('./pages/inference.page').then((m) => m.InferencePage) },
      { path: 'dataset', loadComponent: () => import('./pages/dataset.page').then((m) => m.DatasetPage) },
      { path: 'training', loadComponent: () => import('./pages/training.page').then((m) => m.TrainingPage) },
      { path: 'dataset/validate/:datasetName', loadComponent: () => import('./pages/dataset-validation.page').then((m) => m.DatasetValidationPage) },
    ],
  },
];
