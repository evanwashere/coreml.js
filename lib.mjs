import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { f16, open, predict, features } = require('./dist/coreml.node');

const dtypes = [null, 'i32', 'f64', 'f16', 'f32'];
const AsyncFunction = (async () => {}).constructor;

function u8(buf, thrw = true) {
  if (buf instanceof Uint8Array) return buf;
  if (ArrayBuffer.isView(buf)) return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  if (buf instanceof ArrayBuffer || buf instanceof SharedArrayBuffer) return new Uint8Array(buf);
  if (thrw) throw new TypeError('must be an instance of (ArrayBuffer | ArrayBufferView | SharedArrayBuffer)');
}

const is = {
  url: x => x instanceof URL,
  array: x => Array.isArray(x),
  int: x => Number.isInteger(x),
  float: x => Number.isFinite(x),
  dtype: x => x !== null && dtypes.includes(x),
  buffer: x => is.ab(x) || ArrayBuffer.isView(x),
  object: x => x !== null && typeof x === 'object',
  string: x => x instanceof String || typeof x === 'string',
  sint: x => is.int(x) || (is.string(x) && is.int(Number(x))),
  sfloat: x => is.float(x) || (is.string(x) && is.float(Number(x))),
  ab: x => x instanceof ArrayBuffer || x instanceof SharedArrayBuffer,
  surl: x => { try { return is.url(x) || !!new URL(x); } catch { return false; } },
  
  raw: x => (
    x !== null
    && 'object' === typeof x
    && 'number' === typeof x.width
    && 'number' === typeof x.height
    && ['rgba', 'argb', 'abgr', 'bgra'].includes(x.format)

    && (
      ArrayBuffer.isView(x.buffer)
      || x.buffer instanceof ArrayBuffer
      || x.buffer instanceof SharedArrayBuffer
    )
  ),
};

function get_shape(array, dtype) {
  let ref = array;
  const shape = [];

  while (true) {
    if (is.array(ref)) (shape.push(ref.length), ref = ref[0]);

    else if (is.buffer(ref)) {
      if (!is.ab(ref)) shape.push(ref.length);

      else {
        let buf;
        if ('i32' === dtype) buf = new Int32Array(ref);
        else if ('f16' === dtype) buf = new Uint16Array(ref);
        else if ('f32' === dtype) buf = new Float32Array(ref);
        else if ('f64' === dtype) buf = new Float64Array(ref);

        shape.push(buf.length);
      }

      break;
    }

    else break;
  }

  return shape;
}

const SAI32 = class SingleArray extends Int32Array {
  #ref = null;

  constructor(ref, ...args) {
    super(...args);
    this.#ref = ref;
    this.dtype = 'i32';
    this.shape = [this.length];
  }

  get ref() { return this.#ref; }
  get(offset) { return this[offset]; }
  set(offset, value) { this[offset] = value; }
  copy_from(array, offset) { return super.set(array, offset); }

  cast(dtype) {
    if (!is.dtype(dtype)) throw new TypeError('dtype must be one of: i32, f16, f32, f64');

    const ref = features.multiarray.cast(this.#ref, dtypes.indexOf(dtype));
    
    let buf;
    if ('i32' === dtype) buf = new SAI32(ref, ref.buffer);
    else if ('f16' === dtype) buf = new SAF16(ref, ref.buffer);
    else if ('f32' === dtype) buf = new SAF32(ref, ref.buffer);
    else if ('f64' === dtype) buf = new SAF64(ref, ref.buffer);

    return buf;
  }
}

const SAF32 = class SingleArray extends Float32Array {
  #ref = null;

  constructor(ref, ...args) {
    super(...args);
    this.#ref = ref;
    this.dtype = 'f32';
    this.shape = [this.length];
  }

  get ref() { return this.#ref; }
  get(offset) { return this[offset]; }
  set(offset, value) { this[offset] = value; }
  copy_from(array, offset) { return super.set(array, offset); }

  cast(dtype) {
    if (!is.dtype(dtype)) throw new TypeError('dtype must be one of: i32, f16, f32, f64');

    const ref = features.multiarray.cast(this.#ref, dtypes.indexOf(dtype));
    
    let buf;
    if ('i32' === dtype) buf = new SAI32(ref, ref.buffer);
    else if ('f16' === dtype) buf = new SAF16(ref, ref.buffer);
    else if ('f32' === dtype) buf = new SAF32(ref, ref.buffer);
    else if ('f64' === dtype) buf = new SAF64(ref, ref.buffer);

    return buf;
  }
}

const SAF64 = class SingleArray extends Float64Array {
  #ref = null;

  constructor(ref, ...args) {
    super(...args);
    this.#ref = ref;
    this.dtype = 'f64';
    this.shape = [this.length];
  }

  get ref() { return this.#ref; }
  get(offset) { return this[offset]; }
  set(offset, value) { return this[offset] = value; }
  copy_from(array, offset) { return super.set(array, offset); }

  cast(dtype) {
    if (!is.dtype(dtype)) throw new TypeError('dtype must be one of: i32, f16, f32, f64');

    const ref = features.multiarray.cast(this.#ref, dtypes.indexOf(dtype));
    
    let buf;
    if ('i32' === dtype) buf = new SAI32(ref, ref.buffer);
    else if ('f16' === dtype) buf = new SAF16(ref, ref.buffer);
    else if ('f32' === dtype) buf = new SAF32(ref, ref.buffer);
    else if ('f64' === dtype) buf = new SAF64(ref, ref.buffer);

    return buf;
  }
}

const SAF16 = class SingleArray extends Uint16Array {
  #ref = null;

  constructor(ref, ...args) {
    super(...args);
    this.#ref = ref;
    this.dtype = 'f16';
    this.shape = [this.length];
  }

  get ref() { return this.#ref; }
  get(offset) { return f16.get(this[offset]); }
  set(offset, value) { return this[offset] = f16.new(value); }
  copy_from(array, offset) { return super.set(array, offset); }

  cast(dtype) {
    if (!is.dtype(dtype)) throw new TypeError('dtype must be one of: i32, f16, f32, f64');

    const ref = features.multiarray.cast(this.#ref, dtypes.indexOf(dtype));
    
    let buf;
    if ('i32' === dtype) buf = new SAI32(ref, ref.buffer);
    else if ('f16' === dtype) buf = new SAF16(ref, ref.buffer);
    else if ('f32' === dtype) buf = new SAF32(ref, ref.buffer);
    else if ('f64' === dtype) buf = new SAF64(ref, ref.buffer);

    return buf;
  }
}

class MultiArray extends Array {
  #ref = null;

  constructor(array, dtype = 'f32') {
    if (!is.dtype(dtype)) throw new TypeError('\'dtype\' must be one of: i32, f16, f32, f64');

    if (is.int(array) || is.float(array)) {
      const ref = features.multiarray.new([1], dtypes.indexOf(dtype));

      let buf;
      if ('i32' === dtype) buf = new SAI32(ref, ref.buffer);
      else if ('f16' === dtype) buf = new SAF16(ref, ref.buffer);
      else if ('f32' === dtype) buf = new SAF32(ref, ref.buffer);
      else if ('f64' === dtype) buf = new SAF64(ref, ref.buffer);

      return (buf.set(0, array), buf);
    }

    if (is.buffer(array)) {
      let buf;
      const ref = features.multiarray.new([array.length], dtypes.indexOf(dtype));

      if ('f16' === dtype) {
        buf = new SAF16(ref, ref.buffer);
        if (array instanceof Uint16Array) buf.copy_from(array);
        else if (is.ab(array)) buf.copy_from(new Uint16Array(array));
        else array.forEach((value, offset) => buf.set(offset, value));
      }

      else if ('i32' === dtype) {
        buf = new SAI32(ref, ref.buffer);
        if (array instanceof Int32Array) buf.copy_from(array);
        else if (is.ab(array)) buf.copy_from(new Int32Array(array));

        else array.forEach(!(array instanceof Uint16Array)
          ? (value, offset) => buf.set(offset, value)
          : (value, offset) => buf.set(offset, f16.get(value))
        );
      }

      else if ('f32' === dtype) {
        buf = new SAF32(ref, ref.buffer);
        if (array instanceof Float32Array) buf.copy_from(array);
        else if (is.ab(array)) buf.copy_from(new Float32Array(array));

        else array.forEach(!(array instanceof Uint16Array)
          ? (value, offset) => buf.set(offset, value)
          : (value, offset) => buf.set(offset, f16.get(value))
        );
      }

      else if ('f64' === dtype) {
        buf = new SAF64(ref, ref.buffer);
        if (array instanceof Float64Array) buf.copy_from(array);
        else if (is.ab(array)) buf.copy_from(new Float64Array(array));

        else array.forEach(!(array instanceof Uint16Array)
          ? (value, offset) => buf.set(offset, value)
          : (value, offset) => buf.set(offset, f16.get(value))
        );
      }

      return buf;
    }

    else if (is.array(array)) {
      const shape = get_shape(array, dtype);

      if (1 === shape.length) {
        const ref = features.multiarray.new(shape, dtypes.indexOf(dtype));

        let buf;
        if ('i32' === dtype) buf = new SAI32(ref, ref.buffer);
        else if ('f16' === dtype) buf = new SAF16(ref, ref.buffer);
        else if ('f32' === dtype) buf = new SAF32(ref, ref.buffer);
        else if ('f64' === dtype) buf = new SAF64(ref, ref.buffer);

        for (let o = 0, length = array.length; o < length; o++) {
          if (is.float(array[o])) buf.set(o, array[o]);
          else throw new TypeError('\'array\' must be an array of numbers');
        }

        return buf;
      }

      else {
        super();
        this.shape = shape;
        this.dtype = dtype;
        this.#ref = features.multiarray.new(shape, dtypes.indexOf(dtype));

        let buf;
        let bufo = 0;
        this.buffer = this.#ref.buffer;
        if ('i32' === dtype) buf = new SAI32(this.#ref, this.#ref.buffer);
        else if ('f16' === dtype) buf = new SAF16(this.#ref, this.#ref.buffer);
        else if ('f32' === dtype) buf = new SAF32(this.#ref, this.#ref.buffer);
        else if ('f64' === dtype) buf = new SAF64(this.#ref, this.#ref.buffer);

        const fill = (sub, top, offset) => {
          const dim = shape[offset];
          const last = offset === shape.length - 1;
          const pre_last = offset === shape.length - 2;
          if (!is.array(top) && !is.buffer(top)) throw new TypeError('\'array\' dimension must be an instance of (Array | ArrayBuffer | ArrayBufferView | SharedArrayBuffer)');

          if (pre_last) for (let o = 0; o < dim; o++) fill(sub, top[o], 1 + offset);
          else if (!last) for (let o = 0; o < dim; o++) { const s = []; sub.push(s); fill(s, top[o], 1 + offset); }

          else {
            if ('i32' === dtype) sub.push(new Int32Array(buf.buffer, 4 * bufo, dim));
            else if ('f16' === dtype) sub.push(new Uint16Array(buf.buffer, 2 * bufo, dim));
            else if ('f32' === dtype) sub.push(new Float32Array(buf.buffer, 4 * bufo, dim));
            else if ('f64' === dtype) sub.push(new Float64Array(buf.buffer, 8 * bufo, dim));

            if (is.array(top)) {
              for (let o = 0; o < dim; o++) {
                if (is.float(top[o])) buf.set(bufo++, top[o]);
                else throw new TypeError('\'array\' must be an array of numbers');
              }
            }

            else if (is.buffer(top)) {
              if ('i32' === dtype && top instanceof Int32Array) buf.copy_from(top, bufo);
              else if ('f16' === dtype && top instanceof Uint16Array) buf.copy_from(top, bufo);
              else if ('f32' === dtype && top instanceof Float32Array) buf.copy_from(top, bufo);
              else if ('f64' === dtype && top instanceof Float64Array) buf.copy_from(top, bufo);

              else if (is.ab(top)) {
                if ('i32' === dtype) buf.copy_from(new Int32Array(top), bufo);
                else if ('f16' === dtype) buf.copy_from(new Uint16Array(top), bufo);
                else if ('f32' === dtype) buf.copy_from(new Float32Array(top), bufo);
                else if ('f64' === dtype) buf.copy_from(new Float64Array(top), bufo);
              }

              else top.forEach(!(top instanceof Uint16Array)
                ? (value, offset) => buf.set(bufo + offset, value)
                : (value, offset) => buf.set(bufo + offset, f16.get(value))
              );

              bufo += dim;
            }

            else throw new TypeError('\'array\' last dimension must be an instance of (Array | ArrayBuffer | ArrayBufferView | SharedArrayBuffer)');
          }
        };

        fill(this, array, 0);
      }
    }

    else throw new TypeError('\'array\' must be an instance of (Array | ArrayBuffer | ArrayBufferView | SharedArrayBuffer)');
  }

  get ref() { return this.#ref; }
  cast(dtype) { return new MultiArray(this, dtype); }
}

export class Model {
  #p = null;
  #b = null;
  #ref = null;

  constructor(path, options = {}) {
    if ('string' !== typeof path) throw new TypeError('path must be a string');
    if ('object' !== typeof options) throw new TypeError('options must be an object');

    options.lpaog ??= true; options.units ??= 'all';
    if ('string' !== typeof options.units) throw new TypeError('options.units must be a string');
    if ('boolean' !== typeof options.lpaog) throw new TypeError('options.lpaog must be a boolean');

    const ref = this.#ref = open(path, options);

    this.#p = new Function('$i', '$f', '$P', '$M', '$F', '$MA', `
      let $_ret = null;
      let $_async = false;

      const $_promsie = new Promise(async (ok, err) => {
        try {
          let $_features = $F.new(${Object.keys(ref.i).length});

          ${Object.keys(ref.i).map((name, offset) => `
            $__f${offset}: {
              ${'invalid' == ref.i[name].type ? ''
                : `if (null == $i.${name}) ${ref.i[name].optional ? `break $__f${offset}` : `throw new TypeError('\\'${name}\\' input is required')`};`
              }

              ${{
                invalid() { return `$F.set($_features, '${name}', $F.invalid.new());`; },
                multiarray() { return `$F.set($_features, '${name}', (new $MA($i.${name}, $M.i.${name}.dtype)).ref);`; },
                string() { return `if ('string' !== typeof $i.${name}) throw new TypeError('\\'${name}\\' input must be a string'); $F.set($_features, '${name}', $F.string.new($i.${name}));`; },
                i64() { return `if ('number' !== typeof $i.${name} || Number.isNaN($i.${name})) throw new TypeError('\\'${name}\\' input must be a number'); $F.set($_features, '${name}', $F.i64.new($i.${name}));`; },
                f64() { return `if ('number' !== typeof $i.${name} || Number.isNaN($i.${name})) throw new TypeError('\\'${name}\\' input must be a number'); $F.set($_features, '${name}', $F.f64.new($i.${name}));`; },

                dict() {
                  const kt = ref.i[name].key;

                  return `
                    if ('object' !== typeof $i.${name}) throw new TypeError('\\'${name}\\' input must be an object');

                    ${{
                      string() {
                        return `
                          for (const k in $i.${name}) {
                            if ('string' !== typeof k) throw new TypeError('\\'${name}\\' input must be an object with string keys');
                            if (Number.isNaN($i.${name}[k]) || 'number' !== typeof $i.${name}[k]) throw new TypeError('\\'${name}\\' input must be an object with number values');
                          }
                        `;
                      },

                      i64() {
                        return `
                          for (const k in $i.${name}) {
                            const n = Number(k);
                            if (Number.isNaN(n) || k !== (n >>> 0).toString()) throw new TypeError('\\'${name}\\' input must be an object with integer keys');
                            if (Number.isNaN($i.${name}[k]) || 'number' !== typeof $i.${name}[k]) throw new TypeError('\\'${name}\\' input must be an object with number values');
                          }
                        `;
                      }
                    }[kt]()}

                    $F.set($_features, '${name}', $F.dict.new($i.${name}, $M.i.${name}));
                  `;
                },

                image() {
                  return `
                    function u8(buf, thrw = true) {
                      if (buf instanceof Uint8Array) return buf;
                      if (ArrayBuffer.isView(buf)) return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
                      if (buf instanceof ArrayBuffer || buf instanceof SharedArrayBuffer) return new Uint8Array(buf);
                      if (thrw) throw new TypeError('must be an instance of (ArrayBuffer | ArrayBufferView | SharedArrayBuffer)');
                    }

                    const is = {
                      url: x => x instanceof URL,
                      string: x => x instanceof String || typeof x === 'string',
                      buffer: x => ArrayBuffer.isView(x) || x instanceof ArrayBuffer || x instanceof SharedArrayBuffer,

                      raw: x => (
                        x !== null
                        && 'object' === typeof x
                        && 'number' === typeof x.width
                        && 'number' === typeof x.height
                        && ['rgba', 'argb', 'abgr', 'bgra'].includes(x.format)

                        && (
                          ArrayBuffer.isView(x.buffer)
                          || x.buffer instanceof ArrayBuffer
                          || x.buffer instanceof SharedArrayBuffer
                        )
                      ),
                    };

                    if (is.string($i.${name})) $F.set($_features, '${name}', $F.image.file($i.${name}, $M.i.${name}));
                    else if (is.buffer($i.${name})) $F.set($_features, '${name}', $F.image.buffer(u8($i.${name}), $M.i.${name}));
                    else if (is.url($i.${name})) ($_async = true, $F.set($_features, '${name}', await $F.image.fetch($i.${name}.toString(), $M.i.${name})));

                    else if (is.raw($i.${name})) {
                      $F.set($_features, '${name}', $F.image.raw(
                        {
                          ...$i.${name},
                          format: ['rgba', 'argb', 'abgr', 'bgra'].indexOf($i.${name}.format),
                        },
                        $M.i.${name},
                      ));
                    }

                    else throw new TypeError('\\'${name}\\' input must be a image');
                  `;
                },
              }[ref.i[name].type]()}
            }
          `).join('\n')}

          const ref = $_ret = {};
          $_features = $P($M, $_features);

          ${Object.keys(ref.o).map((name, offset) => `
            $__o${offset}: {
              ${{
                i64() { return `ref.${name} = $F.i64.get($F.get($_features, '${name}'));`; },
                f64() { return `ref.${name} = $F.f64.get($F.get($_features, '${name}'));`; },
                string() { return `ref.${name} = $F.string.get($F.get($_features, '${name}'));`; },
                invalid() { return `ref.${name} = $F.invalid.get($F.get($_features, '${name}'));`; },
                dict() { return `ref.${name} = $F.dict.get($F.get($_features, '${name}'), $M.o.${name});`; },

                image() {
                  return `
                    let format = $f.output?.${name}?.format ?? $f.image.format;
                    if (!['raw', 'png', 'jpg', 'rgba', 'heif', 'tiff'].includes(format))
                      throw new TypeError('flags.output.${name}.format must be one of: raw, png, jpg, rgba, heif, tiff');
                    ref.${name} = $F.image.get($F.get($_features, '${name}'), ['raw', 'rgba', 'jpg', 'png', 'heif', 'tiff'].indexOf(format));
                  `;
                },

                multiarray() {
                  return `
                    const arr = [];
                    const cast = $f.output?.${name}?.dtype;
                    if (cast && !['i32', 'f16', 'f32', 'f64'].includes(cast)) throw new TypeError('flags.output.${name}.cast must be one of: i32, f16, f32, f64');

                    let ma = $F.get($_features, '${name}');
                    if (cast) ma = $F.multiarray.cast(ma, [null, 'i32', 'f64', 'f16', 'f32'].indexOf(cast));

                    ma = $F.multiarray.get(ma);

                    let boffset = 0;
                    arr.dtype = ma.type;
                    arr.shape = ma.shape;
                    arr.buffer = ma.buffer;

                    const fill = (sub, offset) => {
                      const dim = ma.shape[offset];
                      const last = offset === ma.shape.length - 1;
                      const pre_last = offset === ma.shape.length - 2;
                      if (pre_last) for (let o = 0; o < dim; o++) fill(sub, 1 + offset);
                      else if (!last) for (let o = 0; o < dim; o++) { const s = []; sub.push(s); fill(s, 1 + offset); }

                      else {
                        if ('i32' === ma.type) (sub.push(new Int32Array(ma.buffer, boffset, dim)), boffset += 4 * dim);
                        if ('f16' === ma.type) (sub.push(new Uint16Array(ma.buffer, boffset, dim)), boffset += 2 * dim);
                        if ('f32' === ma.type) (sub.push(new Float32Array(ma.buffer, boffset, dim)), boffset += 4 * dim);
                        if ('f64' === ma.type) (sub.push(new Float64Array(ma.buffer, boffset, dim)), boffset += 8 * dim);
                      }
                    };

                    fill(arr, 0);
                    ref.${name} = arr;
                  `;
                },
              }[ref.o[name].type]()}
            }
          `).join('\n')}

          ok(ref);
        } catch (e) {
          err(e);
          $_ret = e;
        }
      });

      if ($_ret instanceof Error) throw $_ret;
      else return !$_async ? $_ret : $_promsie;
    `);
  }

  get path() { return this.#ref.path; }
  get units() { return this.#ref.units; }
  get lpaog() { return this.#ref.lpaog; }
  get input() { return Object.freeze(this.#ref.i); }
  get output() { return Object.freeze(this.#ref.o); }

  predict(i, flags = {}) {
    if ('object' !== typeof i) throw new TypeError('input must be an object');
    if ('object' !== typeof flags) throw new TypeError('flags must be an object');
    if ('object' !== typeof (flags.image ??= {})) throw new TypeError('flags.image must be an object');

    flags.image.format ??= 'raw';
    if ('string' !== typeof flags.image.format) throw new TypeError('flags.image.format must be a string');
    if (!['raw', 'png', 'jpg', 'rgba', 'heif', 'tiff'].includes(flags.image.format)) throw new TypeError('flags.image.format must be one of: raw, png, jpg, rgba, heif, tiff');

    return this.#p(i, flags, predict, this.#ref, features, MultiArray);
  }

  async batch(i, flags = {}) {
    if (!is.array(i)) throw new TypeError('input must be an array');
    if ('object' !== typeof flags) throw new TypeError('flags must be an object');
    if (!i.every(x => is.object(x))) throw new TypeError('input must be an array of objects');
    if ('object' !== typeof (flags.image ??= {})) throw new TypeError('flags.image must be an object');

    flags.image.format ??= 'raw';
    if ('string' !== typeof flags.image.format) throw new TypeError('flags.image.format must be a string');
    if (!['raw', 'png', 'jpg', 'rgba', 'heif', 'tiff'].includes(flags.image.format)) throw new TypeError('flags.image.format must be one of: raw, png, jpg, rgba, heif, tiff');

    return await Promise.all(i.map(x => this.#p(x, flags, predict, this.#ref, features, MultiArray)));
  }
}