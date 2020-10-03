import {Component, OnInit} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Observable} from 'rxjs';
import {map} from 'rxjs/operators';

@Component({
  selector: 'app-food-detail',
  templateUrl: './food-detail.component.html',
  styleUrls: ['./food-detail.component.scss'],
})
export class FoodDetailComponent implements OnInit {
  price: Observable<string>;
  size: Observable<string>;

  constructor(private route: ActivatedRoute) {
  }

  ngOnInit(): void {
    this.price = this.route.paramMap.pipe(map(params => params.get('price')));
    this.size = this.route.queryParamMap.pipe(map(params => params.get('size')));
  }
}
