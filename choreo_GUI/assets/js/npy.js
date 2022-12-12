
class npyjs {

    constructor(opts) {
        if (opts) {
            console.error([
                "No arguments accepted to npyjs constructor.",
                "For usage, go to https://github.com/jhuapl-boss/npyjs."
            ].join(" "));
        }

        this.dtypes = {
            "<u1": {
                name: "uint8",
                size: 8,
                arrayConstructor: Uint8Array,
            },
            "|u1": {
                name: "uint8",
                size: 8,
                arrayConstructor: Uint8Array,
            },
            "<u2": {
                name: "uint16",
                size: 16,
                arrayConstructor: Uint16Array,
            },
            "|i1": {
                name: "int8",
                size: 8,
                arrayConstructor: Int8Array,
            },
            "<i2": {
                name: "int16",
                size: 16,
                arrayConstructor: Int16Array,
            },
            "<u4": {
                name: "uint32",
                size: 32,
                arrayConstructor: Int32Array,
            },
            "<i4": {
                name: "int32",
                size: 32,
                arrayConstructor: Int32Array,
            },
            "<u8": {
                name: "uint64",
                size: 64,
                arrayConstructor: BigUint64Array,
            },
            "<i8": {
                name: "int64",
                size: 64,
                arrayConstructor: BigInt64Array,
            },
            "<f4": {
                name: "float32",
                size: 32,
                arrayConstructor: Float32Array
            },
            "<f8": {
                name: "float64",
                size: 64,
                arrayConstructor: Float64Array
            },
        };

    }

    parse(arrayBufferContents) {
        // const version = arrayBufferContents.slice(6, 8); // Uint8-encoded
        const headerLength = new DataView(arrayBufferContents.slice(8, 10)).getUint8(0);
        const offsetBytes = 10 + headerLength;

        const hcontents = new TextDecoder("utf-8").decode(
            new Uint8Array(arrayBufferContents.slice(10, 10 + headerLength))
        );
        const header = JSON.parse(
            hcontents
                .toLowerCase() // True -> true
                .replace(/'/g, '"')
                .replace("(", "[")
                .replace(/,*\),*/g, "]")
        );
        const shape = header.shape;
        const dtype = this.dtypes[header.descr];
        const nums = new dtype["arrayConstructor"](
            arrayBufferContents,
            offsetBytes
        );
        return {
            dtype: dtype.name,
            data: nums,
            shape,
            fortranOrder: header.fortran_order
        };
    }

    async load(filename, callback, fetchArgs) {
        /*
        Loads an array from a stream of bytes.
        */
        fetchArgs = fetchArgs || {};
        const resp = await fetch(filename, { ...fetchArgs });
        const arrayBuf = await resp.arrayBuffer();
        const result = this.parse(arrayBuf);
        if (callback) {
            return callback(result);
        }
        return result;
    }

// 
//     loadFile(the_file, callback) {
//         /*
//         Loads an array from a File.
//         */
//         const reader = new FileReader()
//         var result
//         var parse = this.parse
//         reader.onload = function() {
//             console.log("aaa",reader.result)
//             result = reader.result
//         }
//         if (callback) {
//             return callback(result)
//         }
//         reader.readAsArrayBuffer(the_file)
//         return  this.parse(result)
// 
//     }

};

const descrToConstructor = {
    "|u1":Uint8Array,
    "|i1":Int8Array,
    "<u2":Uint16Array,
    "<i2":Int16Array,
    "<u4":Uint32Array,
    "<i4":Int32Array,
    "<f4":Float32Array,
    "<f8":Float64Array,
  }
  
const constructorNameToDescr = Object.fromEntries(Object.entries(descrToConstructor).map(x=>[x[1].name,x[0]]));
constructorNameToDescr["Uint8ClampedArray"]="|u1"

const constructorNameToNumBytes = {
    "Uint8Array":1,
    "Int8Array":1,
    "Uint16Array":2,
    "Int16Array":2,
    "Uint32Array":4,
    "Int32Array":4,
    "Float32Array":4,
    "Float64Array":8,
  }

async function ndarray_tobuffer(ndarray){

    var data = ndarray.data;
    var shape = ndarray.shape;
    var Typ = data.constructor
    var dtype_bytes = constructorNameToNumBytes[Typ.name];

    fortran_order = false // HARDCODED HERE
  
    var headerStr = `{'descr': '${constructorNameToDescr[data.constructor.name]}', 'fortran_order': ${['False','True'][Number(fortran_order)]}, 'shape': (${shape.join(", ")},), } `
    
    // 64-byte alignment requirement
    var p = 0; while ((headerStr.length+10+p) % 64 != 0){p += 1;}
  
    var headlen = headerStr.length+p;
    var metalen = headlen+10;
  
    // entire buffer contianing meta info and the data
    var buf = new ArrayBuffer(metalen+data.length*dtype_bytes);
  
    var view = new DataView(buf);
  
    //magic
    view.setUint8(0,147); // \x93
    view.setUint8(1,78); // N
    view.setUint8(2,85); // U
    view.setUint8(3,77); // M
    view.setUint8(4,80); // P
    view.setUint8(5,89); // Y
  
    //version
    view.setUint8(6,1);
    view.setUint8(7,0);
  
    //HEADER_LEN (little endian)
    var n = ((headlen << 8) & 0xFF00) | ((headlen >> 8) & 0xFF)
    view.setUint16(8,n);
  
    for (var i = 0; i < headlen; i++){
      if (i < headerStr.length){
        view.setUint8(10+i, headerStr.charCodeAt(i));
      }else if (i == headlen-1){
        view.setUint8(10+i, 0x0a); //newline terminated
      }else{
        view.setUint8(10+i, 0x20); //space pad
      }
    }
    // pretend the entire buffer is the same type as the TypedArray
    // and modify the underlying data
    new Typ(buf).set(data,metalen/dtype_bytes);
  
    return buf;
  
  }


