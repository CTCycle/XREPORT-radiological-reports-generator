import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./components/main-layout.component').then((m) => m.MainLayoutComponent),
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'inference' },
      { path: 'inference', loadComponent: () => import('./pages/inference.page').then((m) => m.InferencePage) },
      { path: 'reports', loadComponent: () => import('./pages/reports.page').then((m) => m.ReportsPage) },
      { path: 'reports/:requestId', loadComponent: () => import('./pages/report-detail.page').then((m) => m.ReportDetailPage) },
      { path: 'settings', loadComponent: () => import('./pages/settings.page').then((m) => m.SettingsPage) },
      { path: '**', redirectTo: 'inference' },
    ],
  },
];
