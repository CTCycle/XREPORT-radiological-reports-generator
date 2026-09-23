import { CommonModule } from '@angular/common';
import { NO_ERRORS_SCHEMA } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { vi } from 'vitest';
import { DatasetPage } from './dataset.page';
import { DatasetApiService } from '../services/dataset-api.service';
import { ValidationApiService } from '../services/validation-api.service';
import { AppStateService } from '../services/app-state.service';
import { JobPollingService } from '../services/job-polling.service';
import { JobsApiService } from '../services/jobs-api.service';
import { DesktopDialogService } from '../services/desktop-dialog.service';

describe('DatasetPage filesystem access', () => {
  const api = {
    getStatus: vi.fn(),
    getNames: vi.fn(),
    browseDirectory: vi.fn(),
    validateImagePath: vi.fn(),
  };
  let appState: AppStateService;

  async function renderPage(allowServerBrowse: boolean) {
    api.getStatus.mockResolvedValue({
      result: { allow_server_browse: allowServerBrowse, has_data: false, row_count: 0 },
      error: null,
    });
    api.getNames.mockResolvedValue({ result: { datasets: [], count: 0 }, error: null });
    api.browseDirectory.mockReset();
    api.validateImagePath.mockReset();
    appState = new AppStateService();
    appState.updateDataset((state) => ({
      ...state,
      dbStatus: {
        allow_server_browse: allowServerBrowse,
        has_data: false,
        row_count: 0,
        message: 'Filesystem access fixture',
      },
    }));

    TestBed.configureTestingModule({
      imports: [DatasetPage],
      providers: [
        { provide: DatasetApiService, useValue: api },
        { provide: ValidationApiService, useValue: {} },
        { provide: AppStateService, useValue: appState },
        { provide: JobPollingService, useValue: {} },
        { provide: JobsApiService, useValue: {} },
        { provide: Router, useValue: {} },
        { provide: DesktopDialogService, useValue: { isTauriSurface: () => false } },
      ],
    });
    TestBed.overrideComponent(DatasetPage, {
      set: {
        imports: [CommonModule, FormsModule],
        schemas: [NO_ERRORS_SCHEMA],
      },
    });

    const fixture = TestBed.createComponent(DatasetPage);
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();
    return fixture;
  }

  beforeEach(() => {
    api.getStatus.mockReset();
    api.getNames.mockReset();
  });

  it('disables the folder control when server browsing is disabled', async () => {
    const fixture = await renderPage(false);
    const folderButton = fixture.nativeElement.querySelector(
      '.row-datasource .upload-card',
    ) as HTMLButtonElement;

    expect(folderButton.disabled).toBe(true);
    expect(folderButton.textContent).toContain('Disabled by server configuration');
    expect(fixture.componentInstance.canBrowse()).toBe(false);
    fixture.destroy();
  });

  it('recovers from a browse error and selects a valid image folder', async () => {
    const fixture = await renderPage(true);
    const page = fixture.componentInstance;
    const folderButton = fixture.nativeElement.querySelector(
      '.row-datasource .upload-card',
    ) as HTMLButtonElement;
    folderButton.click();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('[role="dialog"]')).not.toBeNull();

    api.browseDirectory
      .mockResolvedValueOnce({ result: null, error: 'Path not found: missing-folder' })
      .mockResolvedValueOnce({
        result: {
          current_path: 'C:\\fixtures\\images',
          parent_path: 'C:\\fixtures',
          items: [],
          drives: ['C:\\'],
        },
        error: null,
      });
    page.browsePath = 'C:\\fixtures\\missing-folder';
    await page.browse();
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Path not found: missing-folder');

    page.browsePath = 'C:\\fixtures\\images';
    await page.browse();
    fixture.detectChanges();
    expect(page.browseError()).toBeNull();

    api.validateImagePath.mockResolvedValue({
      result: {
        valid: true,
        folder_path: 'C:\\fixtures\\images',
        image_count: 1,
        message: 'Found 1 valid images',
      },
      error: null,
    });
    await page.selectFolder();
    fixture.detectChanges();

    expect(fixture.nativeElement.querySelector('[role="dialog"]')).toBeNull();
    expect(fixture.nativeElement.textContent).toContain('images');
    expect(fixture.nativeElement.textContent).toContain('1 images');
    expect(api.validateImagePath).toHaveBeenCalledWith('C:\\fixtures\\images');
    fixture.destroy();
  });
});
