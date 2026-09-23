import { vi } from 'vitest';

import { DesktopDialogService } from './desktop-dialog.service';

const invokeMock = vi.fn();

describe('DesktopDialogService', () => {
  let originalInternalsDescriptor: PropertyDescriptor | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('isTauri', false);

    originalInternalsDescriptor = Object.getOwnPropertyDescriptor(window, '__TAURI_INTERNALS__');
    Object.defineProperty(window, '__TAURI_INTERNALS__', {
      configurable: true,
      value: { invoke: invokeMock },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();

    if (originalInternalsDescriptor) {
      Object.defineProperty(window, '__TAURI_INTERNALS__', originalInternalsDescriptor);
    } else {
      Reflect.deleteProperty(window, '__TAURI_INTERNALS__');
    }
  });

  it('returns a selected native folder path and sends the expected dialog options', async () => {
    vi.stubGlobal('isTauri', true);
    invokeMock.mockResolvedValue('C:\\fixtures\\images');
    const service = new DesktopDialogService();

    await expect(service.openImageFolder()).resolves.toBe('C:\\fixtures\\images');
    expect(invokeMock).toHaveBeenCalledWith('plugin:dialog|open', {
      options: {
        directory: true,
        multiple: false,
        title: 'Select image folder',
      },
    }, undefined);
  });

  it('keeps cancellation as null', async () => {
    vi.stubGlobal('isTauri', true);
    invokeMock.mockResolvedValue(null);
    const service = new DesktopDialogService();

    await expect(service.openImageFolder()).resolves.toBeNull();
    expect(invokeMock).toHaveBeenCalledTimes(1);
  });

  it('does not invoke the Tauri bridge in browser mode', async () => {
    const service = new DesktopDialogService();

    await expect(service.openImageFolder()).resolves.toBeNull();
    expect(invokeMock).not.toHaveBeenCalled();
  });
});
