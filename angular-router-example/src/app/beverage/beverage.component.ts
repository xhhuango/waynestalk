import {Component, OnInit} from '@angular/core';
import {Observable} from 'rxjs';
import {ActivatedRoute} from '@angular/router';
import {map} from 'rxjs/operators';

@Component({
  selector: 'app-beverage',
  templateUrl: './beverage.component.html',
  styleUrls: ['./beverage.component.scss'],
})
export class BeverageComponent implements OnInit {
  name: Observable<string>;

  constructor(private route: ActivatedRoute) {}

  ngOnInit(): void {
    this.name = this.route.paramMap.pipe(map(params => params.get('name')));
  }
}
