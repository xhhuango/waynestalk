import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {FoodComponent} from './food.component';
import {FoodDetailComponent} from './food-detail/food-detail.component';

const routes: Routes = [
  {
    path: 'food',
    component: FoodComponent,
    children: [
      {
        path: 'detail',
        component: FoodDetailComponent,
      },
    ],
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class FoodRoutingModule {
}
