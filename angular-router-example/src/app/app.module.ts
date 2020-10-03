import {BrowserModule} from '@angular/platform-browser';
import {NgModule} from '@angular/core';

import {AppRoutingModule} from './app-routing.module';
import {AppComponent} from './app.component';
import {FoodComponent} from './food/food.component';
import {FoodDetailComponent} from './food/food-detail/food-detail.component';
import {BeverageComponent} from './beverage/beverage.component';
import {FoodModule} from './food/food.module';
import { BannerComponent } from './banner/banner.component';

@NgModule({
  declarations: [
    AppComponent,
    FoodComponent,
    FoodDetailComponent,
    BeverageComponent,
    BannerComponent,
  ],
  imports: [
    BrowserModule,
    FoodModule,
    AppRoutingModule,
  ],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {
}
