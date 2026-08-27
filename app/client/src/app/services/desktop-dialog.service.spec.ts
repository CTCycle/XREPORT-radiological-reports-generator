import { vi } from 'vitest';

const { isTauriMock, openMock } = vi.hoisted(() => ({
  isTauriMock: vi.fn(),
  openMock: vi.fn(),
}));

vi.mock('@tauri-apps/api/core', () => ({ isTauri: isTauriMock }));
vi.mock('@tauri-apps/plugin-dialog', () => ({ open: openMock }));

import { DesktopDialogService } from './desktop-dialog.service';

describe('DesktopDialogService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns a selected native folder path on the Tauri surface', async () => {
    isTauriMock.mockReturnValue(true);
    openMock.mockResolvedValue('C:\\fixtures\\images');
    const service = new DesktopDialogService();

    await expect(service.openImageFolder()).resolves.toBe('C:\\fixtures\\images');
    expect(openMock).toHaveBeenCalledWith({
      directory: true,
      multiple: false,
      title: 'Select image folder',
    });
  });

  it('keeps cancellation as null and does not open the plugin in browser mode', async () => {
    isTauriMock.mockReturnValueOnce(true).mockReturnValueOnce(false);
    openMock.mockResolvedValue(null);
    const service = new DesktopDialogService();

    await expect(service.openImageFolder()).resolves.toBeNull();
    await expect(service.openImageFolder()).resolves.toBeNull();
    expect(openMock).toHaveBeenCalledTimes(1);
  });
});
