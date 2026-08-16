import { CommonModule } from '@angular/common';
import { Component, EventEmitter, HostListener, Input, OnChanges, Output, SimpleChanges, computed, inject, signal } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideAlertCircle, lucideChevronLeft, lucideChevronRight, lucideLoaderCircle, lucideX } from '@ng-icons/lucide';
import { ApiService } from '../services/api.service';
import type { ImageMetadataResponse } from '../types/trainingApi';
import { ModalFocusDirective } from './modal-focus.directive';

@Component({
  standalone: true,
  selector: 'app-image-viewer',
  imports: [CommonModule, NgIcon, ModalFocusDirective],
  providers: [provideIcons({ lucideAlertCircle, lucideChevronLeft, lucideChevronRight, lucideLoaderCircle, lucideX })],
  template: `
    @if (open && datasetName) {
      <div class="modal-backdrop" role="presentation" (click)="closed.emit()">
        <section class="viewer-modal" role="dialog" aria-modal="true" aria-labelledby="image-viewer-title" appModalFocus (modalEscape)="closed.emit()" (click)="$event.stopPropagation()">
          <header class="viewer-header">
            <div class="viewer-title"><h3 id="image-viewer-title">Image Viewer</h3><p class="viewer-subtitle">Dataset: <strong>{{ datasetName }}</strong>@if (totalImages()) { <span class="viewer-counter"> • {{ currentIndex() }} / {{ totalImages() }}</span> }</p></div>
            <button type="button" class="viewer-close" aria-label="Close viewer" (click)="closed.emit()"><ng-icon name="lucideX"/></button>
          </header>
          <div class="viewer-content">
            @if (loadingCount()) { <div class="viewer-loading"><ng-icon name="lucideLoaderCircle" class="spin"/><p>Loading dataset info...</p></div> }
            @else if (error()) { <div class="viewer-error"><ng-icon name="lucideAlertCircle"/><p>{{ error() }}</p></div> }
            @else if (!totalImages()) { <div class="viewer-empty"><p>No images found in this dataset.</p></div> }
            @else {
              <div class="viewer-main">
                <button type="button" class="nav-btn nav-prev" aria-label="Previous image" [disabled]="currentIndex() <= 1" (click)="previous()"><ng-icon name="lucideChevronLeft"/></button>
                <div class="image-display-area">
                  @if (loadingImage() && !imageError()) { <div class="image-loader"><ng-icon name="lucideLoaderCircle" class="spin"/></div> }
                  @if (imageError()) { <div class="image-error-display"><ng-icon name="lucideAlertCircle"/><p>{{ imageError() }}</p></div> }
                  @else if (imageUrl()) { <img class="viewer-image" [src]="imageUrl()" [alt]="metadata()?.image_name || 'X-ray'" [style.display]="loadingImage() ? 'none' : 'block'" (load)="loadingImage.set(false)" (error)="handleImageError()"/> }
                </div>
                <button type="button" class="nav-btn nav-next" aria-label="Next image" [disabled]="currentIndex() >= totalImages()" (click)="next()"><ng-icon name="lucideChevronRight"/></button>
              </div>
            }
          </div>
          @if (!error() && metadata()) { <footer class="viewer-footer"><div class="caption-container"><h4>Radiological Report</h4><p>{{ metadata()?.caption || 'No caption available' }}</p><div class="image-meta"><small>Filename: {{ metadata()?.image_name }}</small></div></div></footer> }
        </section>
      </div>
    }
  `,
  styleUrls: ['../styles/ImageViewerModal.css'],
})
export class ImageViewerComponent implements OnChanges {
  @Input() open = false;
  @Input() datasetName: string | null = null;
  @Output() readonly closed = new EventEmitter<void>();
  private readonly api = inject(ApiService);
  readonly currentIndex = signal(1);
  readonly totalImages = signal(0);
  readonly loadingCount = signal(false);
  readonly loadingImage = signal(false);
  readonly error = signal<string | null>(null);
  readonly metadata = signal<ImageMetadataResponse | null>(null);
  readonly imageError = signal<string | null>(null);
  readonly imageUrl = computed(() => this.datasetName && this.totalImages() > 0 ? this.api.getDatasetImageContentUrl(this.datasetName, this.currentIndex()) : null);

  ngOnChanges(changes: SimpleChanges) {
    if ((changes['open']?.currentValue || changes['datasetName']) && this.open && this.datasetName) void this.loadDataset(this.datasetName);
  }

  private async loadDataset(dataset: string) {
    this.currentIndex.set(1); this.totalImages.set(0); this.metadata.set(null); this.error.set(null); this.imageError.set(null); this.loadingCount.set(true);
    const result = await this.api.getDatasetImageCount(dataset);
    if (!this.open || this.datasetName !== dataset) return;
    this.loadingCount.set(false);
    if (!result.result) { this.error.set(result.error ?? 'Unable to load dataset image count.'); return; }
    this.totalImages.set(result.result.count);
    if (result.result.count > 0) void this.loadMetadata();
  }

  private async loadMetadata() {
    if (!this.datasetName || !this.totalImages()) return;
    this.loadingImage.set(true); this.imageError.set(null);
    const result = await this.api.getDatasetImageMetadata(this.datasetName, this.currentIndex());
    if (!this.open) return;
    if (!result.result) { this.error.set(result.error ?? 'Unable to load image metadata.'); this.loadingImage.set(false); return; }
    this.metadata.set(result.result);
    this.loadingImage.set(false);
    if (!result.result.valid_path) this.imageError.set(`Source file not found at ${result.result.path}`);
  }

  previous() { if (this.currentIndex() > 1) { this.currentIndex.update((index) => index - 1); void this.loadMetadata(); } }
  next() { if (this.currentIndex() < this.totalImages()) { this.currentIndex.update((index) => index + 1); void this.loadMetadata(); } }
  handleImageError() { this.loadingImage.set(false); this.imageError.set(`Failed to load image from ${this.metadata()?.path || 'server'}`); }

  @HostListener('window:keydown', ['$event'])
  handleKeyboard(event: KeyboardEvent) {
    if (!this.open) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); this.previous(); }
    if (event.key === 'ArrowRight') { event.preventDefault(); this.next(); }
  }
}
