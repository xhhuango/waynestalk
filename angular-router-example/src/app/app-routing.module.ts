import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {FoodComponent} from './food/food.component';
import {BeverageComponent} from './beverage/beverage.component';
import {BannerComponent} from './banner/banner.component';

const routes: Routes = [
  {
    path: 'beverage/:name',
    component: BeverageComponent,
  },
  {
    path: 'banner',
    component: BannerComponent,
    outlet: 'ads',
  },
  {
    path: '',
    redirectTo: 'food',
    pathMatch: 'full',
  },
  {
    path: '**',
    component: FoodComponent,
  },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {
}
