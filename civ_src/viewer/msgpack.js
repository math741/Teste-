// Decodificador MessagePack mínimo e completo para os tipos usados pelo protocolo genesis-live.
// bin → Uint8Array (cópia alinhada, pronta para TypedArray), str → string, map → objeto, array → array.
const td = new TextDecoder();

export function decode(buf) {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let p = 0;
  function bin(n) { const out = new Uint8Array(n); out.set(u8.subarray(p, p + n)); p += n; return out; }
  function str(n) { const s = td.decode(u8.subarray(p, p + n)); p += n; return s; }
  function arr(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = read(); return a; }
  function map(n) { const o = {}; for (let i = 0; i < n; i++) { const k = read(); o[k] = read(); } return o; }
  function read() {
    const b = u8[p++];
    if (b <= 0x7f) return b;
    if (b >= 0xe0) return b - 0x100;
    if ((b & 0xf0) === 0x80) return map(b & 0x0f);
    if ((b & 0xf0) === 0x90) return arr(b & 0x0f);
    if ((b & 0xe0) === 0xa0) return str(b & 0x1f);
    let v;
    switch (b) {
      case 0xc0: return null;
      case 0xc2: return false;
      case 0xc3: return true;
      case 0xc4: v = u8[p]; p += 1; return bin(v);
      case 0xc5: v = dv.getUint16(p); p += 2; return bin(v);
      case 0xc6: v = dv.getUint32(p); p += 4; return bin(v);
      case 0xca: v = dv.getFloat32(p); p += 4; return v;
      case 0xcb: v = dv.getFloat64(p); p += 8; return v;
      case 0xcc: return u8[p++];
      case 0xcd: v = dv.getUint16(p); p += 2; return v;
      case 0xce: v = dv.getUint32(p); p += 4; return v;
      case 0xcf: v = Number(dv.getBigUint64(p)); p += 8; return v;
      case 0xd0: v = dv.getInt8(p); p += 1; return v;
      case 0xd1: v = dv.getInt16(p); p += 2; return v;
      case 0xd2: v = dv.getInt32(p); p += 4; return v;
      case 0xd3: v = Number(dv.getBigInt64(p)); p += 8; return v;
      case 0xd9: v = u8[p]; p += 1; return str(v);
      case 0xda: v = dv.getUint16(p); p += 2; return str(v);
      case 0xdb: v = dv.getUint32(p); p += 4; return str(v);
      case 0xdc: v = dv.getUint16(p); p += 2; return arr(v);
      case 0xdd: v = dv.getUint32(p); p += 4; return arr(v);
      case 0xde: v = dv.getUint16(p); p += 2; return map(v);
      case 0xdf: v = dv.getUint32(p); p += 4; return map(v);
      case 0xd4: p += 2; return null;            // fixext (não usado): ignora com segurança
      case 0xd5: p += 3; return null;
      case 0xd6: p += 5; return null;
      case 0xd7: p += 9; return null;
      case 0xd8: p += 17; return null;
      case 0xc7: v = u8[p]; p += 2 + v; return null;
      case 0xc8: v = dv.getUint16(p); p += 3 + v; return null;
      case 0xc9: v = dv.getUint32(p); p += 5 + v; return null;
      default: throw new Error('msgpack: byte inválido 0x' + b.toString(16));
    }
  }
  return read();
}

// Visões tipadas (little-endian, como numpy no servidor)
export const asU16 = (b) => new Uint16Array(b.buffer, b.byteOffset, b.byteLength >> 1);
export const asI16 = (b) => new Int16Array(b.buffer, b.byteOffset, b.byteLength >> 1);
export const asF32 = (b) => new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2);
export const asI64 = (b) => new BigInt64Array(b.buffer, b.byteOffset, b.byteLength >> 3);
export function f16(b) {
  const u = asU16(b), out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}
