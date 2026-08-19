import { StorageService } from './storage.service';

describe('StorageService', () => {
  beforeEach(() => localStorage.clear());

  it('round-trips persisted records', () => {
    const service = new StorageService();
    service.writeRecord('jobs', { first: { status: 'running' } });
    expect(service.readRecord<{ status: string }>('jobs')).toEqual({ first: { status: 'running' } });
  });

  it('round-trips scalar values', () => {
    const service = new StorageService();
    service.writeString('theme-preference', 'dark');
    expect(service.readString('theme-preference')).toBe('dark');
  });

  it('recovers from malformed storage without throwing', () => {
    localStorage.setItem('jobs', '{not-json');
    expect(new StorageService().readRecord('jobs')).toEqual({});
  });
});
