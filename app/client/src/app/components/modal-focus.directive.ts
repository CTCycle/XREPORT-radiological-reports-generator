import { FocusTrap, FocusTrapFactory } from '@angular/cdk/a11y';
import { AfterViewInit, Directive, ElementRef, EventEmitter, HostListener, OnDestroy, Output, inject } from '@angular/core';

@Directive({
  selector: '[appModalFocus]',
  standalone: true,
})
export class ModalFocusDirective implements AfterViewInit, OnDestroy {
  private readonly element = inject(ElementRef<HTMLElement>);
  private readonly trapFactory = inject(FocusTrapFactory);
  private trap: FocusTrap | null = null;
  @Output() readonly modalEscape = new EventEmitter<KeyboardEvent>();

  ngAfterViewInit() {
    this.trap = this.trapFactory.create(this.element.nativeElement, true);
    this.trap.focusInitialElementWhenReady();
  }

  @HostListener('keydown.escape', ['$event'])
  handleEscape(event: Event) {
    event.preventDefault();
    this.modalEscape.emit(event as KeyboardEvent);
  }

  ngOnDestroy() {
    this.trap?.destroy();
    this.trap = null;
  }
}
