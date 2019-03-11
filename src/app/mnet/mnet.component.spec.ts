import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { MnetComponent } from './mnet.component';

describe('MnetComponent', () => {
  let component: MnetComponent;
  let fixture: ComponentFixture<MnetComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ MnetComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(MnetComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
