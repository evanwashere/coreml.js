import { Model } from '../lib.mjs';

const model = new Model(new URL('mnist.mlpackage', import.meta.url).pathname);

console.log(model.input);
console.log(model.output);

console.log(await model.predict({
  image: new URL('https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Numerical_digit_1_in_red_circle.svg/640px-Numerical_digit_1_in_red_circle.svg.png'),
}));