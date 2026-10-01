import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { ModalFocusDirective } from './modal-focus.directive';

@Component({
  standalone: true,
  imports: [ModalFocusDirective],
  template: '<button type="button">Open dialog</button><section role="dialog" appModalFocus><button>First action</button><button>Last action</button></section>',
})
class ModalFocusHostComponent {}

describe('ModalFocusDirective', () => {
  it('attaches boundary anchors for modal focus trapping', async () => {
    await TestBed.configureTestingModule({ imports: [ModalFocusHostComponent] }).compileComponents();
    const fixture = TestBed.createComponent(ModalFocusHostComponent);
    fixture.detectChanges();
    await fixture.whenStable();

    const dialog = fixture.nativeElement.querySelector('[role="dialog"]') as HTMLElement;
    const anchors = dialog.parentElement?.querySelectorAll('.cdk-focus-trap-anchor');

    expect(anchors?.length).toBe(2);
    fixture.destroy();
  });
});
