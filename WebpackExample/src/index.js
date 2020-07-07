import _ from 'lodash';
import './style.css';
import Icon from './icon.png';

const app = document.getElementById('app');
app.innerHTML = _.join(['Hello', 'Webpack'], ' ');
app.classList.add('hello');

const icon = new Image();
icon.src = Icon;
app.appendChild(icon);

