<h1 align=center>coreml.js</h1>
<div align=center>cross-runtime library for CoreML</div>
<br />

### Install
`bun add coreml`

`npm install coreml`

## Example
```js
import { Model } from 'coreml';

const model = new Model('mnist.mlpackage', { units: 'all' });

// inspect input and output types
console.log(model.input);
console.log(model.output);

model.predict({
  image: 'one.jpg',
});

/* -> {
  labelProbabilities: {
    '0': 0.0023670196533203125,
    '1': 0.96533203125,
    '2': 5.960464477539063e-8,
    '3': 0,
    '4': 1.7881393432617188e-7,
    '5': 0.0002435445785522461,
    '6': 0.032470703125,
    '7': 8.940696716308594e-7,
    '8': 1.7881393432617188e-7,
    '9': 0
  },
  classLabel: 1n
} */
```


## License

MIT © [Evan](https://github.com/evanwashere)