import {Component} from '@angular/core';
import {ActivatedRoute, Router} from '@angular/router';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent {
  title = 'angular-router-example';

  constructor(private router: Router,
              private route: ActivatedRoute) {
  }

  show7Up(): void {
    this.router.navigate(['beverage', '7UP'], {relativeTo: this.route});
  }
}
