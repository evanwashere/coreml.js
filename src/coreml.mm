#include <vector>
#include <string>
#include <variant>
#include <stdexcept>
#include <functional>
#include "node_api.mm"
#include <CoreML/CoreML.h>
#include <CoreImage/CoreImage.h>
#include <Foundation/Foundation.h>

void init(napi_env env, napi_value exports);
NAPI_MODULE_INIT() { init(env, exports); return exports; }

napi::value ModelDescription(napi::env env, NSDictionary<NSString*, MLFeatureDescription*> *descriptions);

void init(napi::env env, napi::value exports) {
  using MLFeatureDictionary = NSMutableDictionary<NSString*, MLFeatureValue*>;

  napi::object::set(env, exports, "f16", napi::object::from(env, {
    {"new", napi::function::from(env, "new", [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto args = napi::function::args(env, info, 1);
        auto f = (_Float16)napi::number::to(env, args[0]);

        return napi::number::from(env, *(uint16_t*)&f);
      }
    })},

    {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto args = napi::function::args(env, info, 1);

        auto i = napi::number::to_u16(env, args[0]);
        return napi::number::from(env, (double)*(_Float16*)&i);
      }
    })},
  }));

  napi::object::set(env, exports, "predict", napi::function::from(env, "predict",
    [](auto env, auto info) -> auto {
      @autoreleasepool {
        NSError *error = nil;
        auto args = napi::function::args(env, info, 2);
        auto model = napi::object::unwrap<MLModel*>(env, args[0]);
        auto in = napi::object::unwrap<MLFeatureDictionary*>(env, args[1]);
        auto features = [[[MLDictionaryFeatureProvider new] autorelease] initWithDictionary: in error: &error];

        if (unlikely(error)) return napi::err(env, error.localizedDescription);
        id<MLFeatureProvider> res = [model predictionFromFeatures: features error: &error];

        auto ref = napi::object::empty(env);
        if (unlikely(error)) return napi::err(env, error.localizedDescription);

        auto names = [res featureNames];
        auto out = [[MLFeatureDictionary dictionaryWithCapacity: names.count] retain];
        for (NSString *name in names) [out setObject: [res featureValueForName: name] forKey: name];
        napi::object::wrap<MLFeatureDictionary*>(env, ref, out, [](auto env, auto ref) { [ref release]; });

        return ref;
      }
    }
  ));

  napi::object::set(env, exports, "batch", napi::function::from(env, "batch",
    [](auto env, auto info) -> auto {
      return napi::null(env);

      @autoreleasepool {
        auto args = napi::function::args(env, info, 2);
        auto model = napi::object::unwrap<MLModel*>(env, args[0]);
        auto batch = [[[NSMutableArray<MLDictionaryFeatureProvider*> new] autorelease] initWithCapacity: napi::array::length(env, args[1])];

        for (auto ref : napi::array::iterator(env, args[1])) {
          NSError *error = nil;
          auto in = napi::object::unwrap<MLFeatureDictionary*>(env, ref);
          auto features = [[[MLDictionaryFeatureProvider new] autorelease] initWithDictionary: in error: &error];
          if (unlikely(error)) return napi::err(env, error.localizedDescription); else [batch addObject: features];
        }

        id<MLBatchProvider> provider = [[[MLArrayBatchProvider new] autorelease] initWithFeatureProviderArray: batch];

        NSError *error = nil;
        id<MLBatchProvider> res = [model predictionsFromBatch: provider error: &error];
        if (unlikely(error)) return napi::err(env, error.localizedDescription); size_t count = [res count];

        auto ref = napi::array::zeroed(env, count);

        for (size_t i = 0; i < count; i++) {
          auto sub = napi::object::empty(env);
          auto names = [[res featuresAtIndex: i] featureNames];
          auto out = [[MLFeatureDictionary dictionaryWithCapacity: names.count] retain];
          for (NSString *name in names) [out setObject: [[res featuresAtIndex: i] featureValueForName: name] forKey: name];

          napi::array::set(env, ref, i, sub);
          napi::object::wrap<MLFeatureDictionary*>(env, sub, out, [](auto env, auto ref) { [ref release]; });
        }

        return ref;
      }
    }
  ));

  napi::object::set(env, exports, "open", napi::function::from(env, "open",
    [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto args = napi::function::args(env, info, 2);
        MLModelConfiguration *configuration = [[MLModelConfiguration new] autorelease];
        NSString *units = napi::string::to(env, napi::object::get(env, args[1], "units"));
        if ([units isEqualToString: @"all"]) configuration.computeUnits = MLComputeUnitsAll;
        else if ([units isEqualToString: @"cpu"]) configuration.computeUnits = MLComputeUnitsCPUOnly;
        else if ([units isEqualToString: @"gpu"]) configuration.computeUnits = MLComputeUnitsCPUAndGPU;
        else if ([units isEqualToString: @"ane"]) configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        configuration.allowLowPrecisionAccumulationOnGPU = napi::boolean::to(env, napi::object::get(env, args[1], "lpaog"));

        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath: napi::string::to(env, args[0]) isDirectory: true];

        if (![url.path hasSuffix: @".mlmodelc"]) {
          url = [MLModel compileModelAtURL: url error: &error];
          if (unlikely(error)) return napi::err(env, error.localizedDescription);
        }

        MLModel *model = [[MLModel modelWithContentsOfURL: url configuration: configuration error: &error] retain];

        if (unlikely(error)) return napi::err(env, error.localizedDescription);
        napi::object::set(env, args[1], "path", napi::string::from(env, url.path));
        napi::object::wrap<MLModel*>(env, args[1], model, [](auto env, auto model) { [model release]; });
        napi::object::set(env, args[1], "i", ModelDescription(env, model.modelDescription.inputDescriptionsByName));
        napi::object::set(env, args[1], "o", ModelDescription(env, model.modelDescription.outputDescriptionsByName));

        return args[1];
      }
    }
  ));

  napi::object::set(env, exports, "features", napi::object::from(env, {
    {"clear", napi::function::from(env, "clear", [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto args = napi::function::args(env, info, 1);
        [napi::object::unwrap<MLFeatureDictionary*>(env, args[0]) removeAllObjects];

        return args[0];
      }
    })},

    {"set", napi::function::from(env, "set", [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto args = napi::function::args(env, info, 3);
        auto value = napi::object::unwrap<MLFeatureValue*>(env, args[2]);
        auto in = napi::object::unwrap<MLFeatureDictionary*>(env, args[0]);
        [in setObject: value forKey: napi::string::to(env, args[1])]; return args[0];
      }
    })},

    {"keys", napi::function::from(env, "keys", [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto args = napi::function::args(env, info, 1);
        auto in = napi::object::unwrap<MLFeatureDictionary*>(env, args[0]);
        auto keys = [in allKeys]; auto ref = napi::array::zeroed(env, keys.count);
        for (size_t o = 0; o < keys.count; o++) napi::array::set(env, ref, o, napi::string::from(env, keys[o]));

        return ref;
      }
    })},

    {"new", napi::function::from(env, "new", [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto ref = napi::object::empty(env);
        auto args = napi::function::args(env, info, 1);
        size_t capacity = napi::number::to_u32(env, args[0]);
        auto in = [[MLFeatureDictionary dictionaryWithCapacity: capacity] retain];
        napi::object::wrap<MLFeatureDictionary*>(env, ref, in, [](auto env, auto ref) { [ref release]; });

        return ref;
      }
    })},

    {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
      @autoreleasepool {
        auto ref = napi::object::empty(env);
        auto args = napi::function::args(env, info, 2);
        auto in = napi::object::unwrap<MLFeatureDictionary*>(env, args[0]);

        auto key = napi::string::to(env, args[1]);
        auto value = [[in objectForKey: key] retain]; if (unlikely(!value)) return napi::null(env);
        napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

        return ref;
      }
    })},

    {"invalid", napi::object::from(env, {
      {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto args = napi::function::args(env, info, 1);
          auto value = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          if ([value type] != MLFeatureTypeInvalid) return napi::err(env, "feature must be an instance of invalid");

          return napi::null(env);
        }
      })},

      {"new", napi::function::from(env, "new", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto value = [[MLFeatureValue undefinedFeatureValueWithType: MLFeatureTypeInvalid] retain];
          napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},
    })},

    {"string", napi::object::from(env, {
      {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto args = napi::function::args(env, info, 1);
          auto value = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          if ([value type] != MLFeatureTypeString) return napi::err(env, "feature must be an instance of string");

          return napi::string::from(env, [value stringValue]);
        }
      })},

      {"new", napi::function::from(env, "new", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 1);
          auto value = [[MLFeatureValue featureValueWithString: napi::string::to(env, args[0])] retain];
          napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},
    })},

    {"i64", napi::object::from(env, {
      {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto args = napi::function::args(env, info, 1);
          auto value = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          if ([value type] != MLFeatureTypeInt64) return napi::err(env, "feature must be an instance of i64");

          return napi::bigint::from(env, [value int64Value]);
        }
      })},

      {"new", napi::function::from(env, "new", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 1);

          auto number = napi_number == napi::type(env, args[0])
            ? napi::number::to_i64(env, args[0]) : napi::bigint::to_i64(env, args[0]);

          auto value = [[MLFeatureValue featureValueWithInt64: number] retain];
          napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},
    })},

    {"f64", napi::object::from(env, {
      {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto args = napi::function::args(env, info, 1);
          auto value = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          if ([value type] != MLFeatureTypeDouble) return napi::err(env, "feature must be an instance of f64");

          return napi::number::from(env, [value doubleValue]);
        }
      })},

      {"new", napi::function::from(env, "new", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 1);

          auto number = napi_number == napi::type(env, args[0])
            ? napi::number::to(env, args[0]) : (double_t)napi::bigint::to_i64(env, args[0]);

          auto value = [[MLFeatureValue featureValueWithDouble: number] retain];
          napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},
    })},

    {"dict", napi::object::from(env, {
      {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);
          auto value = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          auto constraint = napi::object::unwrap<MLDictionaryConstraint*>(env, args[1]);
          if ([value type] != MLFeatureTypeDictionary) return napi::err(env, "feature must be an instance of dict");

          auto dict = [value dictionaryValue];

          if (MLFeatureTypeString == constraint.keyType) for (NSString *key in [dict allKeys]) {
            auto value = [dict objectForKey: key];
            napi::object::set(env, ref, key, napi::number::from(env, [value doubleValue]));
          }

          else if (MLFeatureTypeInt64 == constraint.keyType) for (NSNumber *key in [dict allKeys]) {
            auto value = [dict objectForKey: key];
            auto k = napi::number::from(env, [key longLongValue]);
            napi::object::set(env, ref, k, napi::number::from(env, [value doubleValue]));
          }

          return ref;
        }
      })},

      {"new", napi::function::from(env, "new", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);
          auto dict = [[NSMutableDictionary<id, NSNumber*> new] autorelease];
          auto keys = napi::array::iterator(env, napi::object::keys(env, args[0]));
          auto constraint = napi::object::unwrap<MLDictionaryConstraint*>(env, args[1]);

          for (auto key : keys) {
            auto value = napi::object::get(env, args[0], key);
            auto v = [NSNumber numberWithDouble: napi::number::to(env, value)];

            if (constraint.keyType == MLFeatureTypeString)
              [dict setObject: v forKey: napi::string::to(env, key)];

            else if (constraint.keyType == MLFeatureTypeInt64) {
              auto k = napi::number::coerce(env, key);
              [dict setObject: v forKey: [NSNumber numberWithLongLong: napi::number::to_i64(env, k)]];
            }
          }

          NSError *error = nil;
          auto value = [[MLFeatureValue featureValueWithDictionary: dict error: &error] retain];

          if (unlikely(error)) return napi::err(env, error.localizedDescription);
          napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},
    })},

    {"multiarray", napi::object::from(env, {
      {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 1);
          auto value = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          if ([value type] != MLFeatureTypeMultiArray) return napi::err(env, "feature must be an instance of multiarray");

          auto arr = [value multiArrayValue];

          NSString *t = nil;
          auto type = [arr dataType];
          size_t length = [arr count];
          auto shape = napi::array::empty(env);

          for (NSNumber *dim in arr.shape) {
            napi::array::push(env, shape, napi::number::from(env, dim.unsignedIntegerValue));
          }

          if (type == MLMultiArrayDataTypeInt32) { t = @"i32", length *= 4; }
          else if (type == MLMultiArrayDataTypeDouble) { t = @"f64", length *= 8; }
          else if (type == MLMultiArrayDataTypeFloat16) { t = @"f16", length *= 2; }
          else if (type == MLMultiArrayDataTypeFloat32) { t = @"f32", length *= 4; }

          napi::object::set(env, ref, "shape", shape);
          napi::object::set(env, ref, "type", napi::string::from(env, t));
          napi::object::set(env, ref, "length", napi::number::from(env, arr.count));

          napi::object::set(env, ref, "buffer", napi::alloc::from(
            env, length, arr.dataPointer, [arr retain],
            [](auto env, auto ptr, auto hint) { [(MLMultiArray*)hint release]; }
          ));

          return ref;
        }
      })},

      {"new", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);

          MLMultiArrayDataType type;
          auto t = napi::number::to_u32(env, args[1]);
          NSMutableArray<NSNumber*> *shape = [[NSMutableArray new] autorelease];

          if (t == 1) type = MLMultiArrayDataTypeInt32;
          else if (t == 2) type = MLMultiArrayDataTypeDouble;
          else if (t == 3) type = MLMultiArrayDataTypeFloat16;
          else if (t == 4) type = MLMultiArrayDataTypeFloat32;

          for (auto dim : napi::array::iterator(env, args[0])) {
            [shape addObject: [NSNumber numberWithUnsignedInteger: napi::number::to_u32(env, dim)]];
          }

          NSError *error = nil;
          auto arr = [[[MLMultiArray new] autorelease] initWithShape: shape dataType: type error: &error];

          if (unlikely(error)) return napi::err(env, error.localizedDescription);
          auto value = [[MLFeatureValue featureValueWithMultiArray: arr] retain];
          napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

          size_t length = arr.count;

          if (type == MLMultiArrayDataTypeInt32) length *= 4;
          else if (type == MLMultiArrayDataTypeDouble) length *= 8;
          else if (type == MLMultiArrayDataTypeFloat16) length *= 2;
          else if (type == MLMultiArrayDataTypeFloat32) length *= 4;

          napi::object::set(env, ref, "buffer", napi::alloc::from(
            env, length, arr.dataPointer, [arr retain],
            [](auto env, auto ptr, auto hint) { [(MLMultiArray*)hint release]; }
          ));

          return ref;
        }
      })},

      {"cast", napi::function::from(env, "cast", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);
          auto value = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          if ([value type] != MLFeatureTypeMultiArray) return napi::err(env, "feature must be an instance of multiarray");

          MLMultiArrayDataType type;
          auto t = napi::number::to_u32(env, args[1]);
          if (t == 1) type = MLMultiArrayDataTypeInt32;
          else if (t == 2) type = MLMultiArrayDataTypeDouble;
          else if (t == 3) type = MLMultiArrayDataTypeFloat16;
          else if (t == 4) type = MLMultiArrayDataTypeFloat32;

          auto arr = [value multiArrayValue];
          arr = [MLMultiArray multiArrayByConcatenatingMultiArrays: @[arr] alongAxis: 0 dataType: type];

          value = [[MLFeatureValue featureValueWithMultiArray: arr] retain];
          napi::object::wrap<MLFeatureValue*>(env, ref, value, [](auto env, auto ref) { [ref release]; });

          size_t length = arr.count;
          auto shape = napi::array::empty(env);
          if (type == MLMultiArrayDataTypeInt32) length *= 4;
          else if (type == MLMultiArrayDataTypeDouble) length *= 8;
          else if (type == MLMultiArrayDataTypeFloat16) length *= 2;
          else if (type == MLMultiArrayDataTypeFloat32) length *= 4;

          for (NSNumber *dim in arr.shape) {
            napi::array::push(env, shape, napi::number::from(env, dim.unsignedIntegerValue));
          }

          napi::object::set(env, ref, "shape", shape);
          napi::object::set(env, ref, "type", napi::number::from(env, t));
          napi::object::set(env, ref, "length", napi::number::from(env, arr.count));

          napi::object::set(env, ref, "buffer", napi::alloc::from(
            env, length, arr.dataPointer, [arr retain],
            [](auto env, auto ptr, auto hint) { [(MLMultiArray*)hint release]; }
          ));

          return ref;
        }
      })},
    })},

    {"image", napi::object::from(env, {
      {"file", napi::function::from(env, "file", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);
          auto constraint = napi::object::unwrap<MLImageConstraint*>(env, args[1]);
          auto url = [NSURL fileURLWithPath: napi::string::to(env, args[0]) isDirectory: false];

          NSError *error = nil;
          auto feature = [[MLFeatureValue featureValueWithImageAtURL: url constraint: constraint options: nil error: &error] retain];

          if (unlikely(error)) return napi::err(env, error.localizedDescription);
          napi::object::wrap<MLFeatureValue*>(env, ref, feature, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},

      {"buffer", napi::function::from(env, "buffer", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto ctx = [CIContext contextWithOptions: nil];
          auto args = napi::function::args(env, info, 2);
          auto constraint = napi::object::unwrap<MLImageConstraint*>(env, args[1]);

          auto [ptr, len] = napi::slice::info(env, args[0]);
          auto image = [CIImage imageWithData: [NSData dataWithBytes: ptr length: len]];

          if (unlikely(!image)) return napi::err(env, "corrupted image");
          auto cg = [ctx createCGImage: image fromRect: CGRectMake(0, 0, image.extent.size.width, image.extent.size.height)];

          NSError *error = nil;
          auto feature = [[MLFeatureValue featureValueWithCGImage: cg constraint: constraint options: nil error: &error] retain];

          CGImageRelease(cg);
          if (unlikely(error)) return napi::err(env, error.localizedDescription);
          napi::object::wrap<MLFeatureValue*>(env, ref, feature, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},

      {"fetch", napi::function::from(env, "fetch", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);
          auto url = [NSURL URLWithString: napi::string::to(env, args[0])];
          auto constraint = napi::object::unwrap<MLImageConstraint*>(env, args[1]);

          struct Info {
            NSURL *url;
            MLImageConstraint *constraint;
          };

          [url retain]; [constraint retain];
          auto info = new Info {url, constraint};

          return napi::async::create<MLFeatureValue*>(env, "fetch", info,
            [](auto ptr) -> std::variant<MLFeatureValue*, std::runtime_error> {
              @autoreleasepool {
                NSError *error = nil;
                auto info = (Info*)ptr;
                auto feature = [[MLFeatureValue featureValueWithImageAtURL: info->url constraint: info->constraint options: nil error: &error] retain];

                [info->url release]; [info->constraint release]; delete info;
                if (unlikely(error)) return std::runtime_error(error.localizedDescription.UTF8String);

                return feature;
              }
            },

            [](auto env, auto feature) -> auto {
              auto ref = napi::object::empty(env);
              napi::object::wrap<MLFeatureValue*>(env, ref, feature, [](auto env, auto ref) { [ref release]; });

              return ref;
            }
          );
        }
      })},

      {"raw", napi::function::from(env, "raw", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);
          auto ctx = [CIContext contextWithOptions: nil];
          auto constraint = napi::object::unwrap<MLImageConstraint*>(env, args[1]);
          size_t width = napi::number::to_u32(env, napi::object::get(env, args[0], "width"));
          auto [ptr, len] = napi::slice::info(env, napi::object::get(env, args[0], "buffer"));
          size_t height = napi::number::to_u32(env, napi::object::get(env, args[0], "height"));
          uint32_t format = napi::number::to_u32(env, napi::object::get(env, args[0], "format"));

          CIFormat f;
          if (format == 0) f = kCIFormatRGBA8;
          else if (format == 1) f = kCIFormatARGB8;
          else if (format == 2) f = kCIFormatABGR8;
          else if (format == 3) f = kCIFormatBGRA8;
          else return napi::err(env, "invalid format");
          auto color_space = CGColorSpaceCreateDeviceRGB();

          auto image = [CIImage
            imageWithBitmapData: [NSData dataWithBytes: ptr length: len]
            bytesPerRow: 4 * width size: CGSizeMake(width, height) format: f colorSpace: color_space
          ];

          if (unlikely(!image)) { CGColorSpaceRelease(color_space); return napi::err(env, "corrupted raw image"); }
          auto cg = [ctx createCGImage: image fromRect: CGRectMake(0, 0, image.extent.size.width, image.extent.size.height)];

          NSError *error = nil;
          auto feature = [[MLFeatureValue featureValueWithCGImage: cg constraint: constraint options: nil error: &error] retain];

          CGImageRelease(cg);
          CGColorSpaceRelease(color_space);
          if (unlikely(error)) return napi::err(env, error.localizedDescription);
          napi::object::wrap<MLFeatureValue*>(env, ref, feature, [](auto env, auto ref) { [ref release]; });

          return ref;
        }
      })},

      {"get", napi::function::from(env, "get", [](auto env, auto info) -> auto {
        @autoreleasepool {
          auto ref = napi::object::empty(env);
          auto args = napi::function::args(env, info, 2);
          auto encoding = napi::number::to_u32(env, args[1]);
          auto feature = napi::object::unwrap<MLFeatureValue*>(env, args[0]);
          if ([feature type] != MLFeatureTypeImage) return napi::err(env, "feature must be an instance of image");

          CVPixelBufferRef image = [feature imageBufferValue];

          auto width = CVPixelBufferGetWidth(image);
          auto height = CVPixelBufferGetHeight(image);
          napi::object::set(env, ref, "width", napi::number::from(env, (uint32_t)width));
          napi::object::set(env, ref, "height", napi::number::from(env, (uint32_t)height));

          if (0 == encoding) { // raw
            auto channels = CVPixelBufferGetBytesPerRow(image) / width;
            CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
            auto ab = napi::alloc::copy(env, width * height * channels, CVPixelBufferGetBaseAddress(image));

            CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
            napi::object::set(env, ref, "buffer", napi::slice::u8(env, ab, 0, width * height * channels));

            OSType type = CVPixelBufferGetPixelFormatType(image);
            if (type == kCVPixelFormatType_32RGBA) napi::object::set(env, ref, "format", napi::string::from(env, "rgba32"));
            else if (type == kCVPixelFormatType_32ARGB) napi::object::set(env, ref, "format", napi::string::from(env, "argb32"));
            else if (type == kCVPixelFormatType_32ABGR) napi::object::set(env, ref, "format", napi::string::from(env, "abgr32"));
            else if (type == kCVPixelFormatType_32BGRA) napi::object::set(env, ref, "format", napi::string::from(env, "bgra32"));
            else if (type == kCVPixelFormatType_OneComponent8) napi::object::set(env, ref, "format", napi::string::from(env, "gray8"));
            else if (type == kCVPixelFormatType_OneComponent16Half) napi::object::set(env, ref, "format", napi::string::from(env, "gray16"));
          }

          else {
            auto ctx = [CIContext contextWithOptions: nil];
            auto color_space = CGColorSpaceCreateDeviceRGB();
            CIImage *ci = [CIImage imageWithCVPixelBuffer: image];

            if (1 == encoding) { // rgba
              auto ab = napi::alloc::zeroed(env, 4 * width * height);
              napi::object::set(env, ref, "format", napi::string::from(env, "rgba32"));
              CGImageRef cg = [ctx createCGImage: ci fromRect: CGRectMake(0, 0, width, height)];
              napi::object::set(env, ref, "buffer", napi::slice::u8(env, ab, 0, 4 * width * height));

              auto bctx = CGBitmapContextCreate(
                napi::alloc::ptr(env, ab), width, height,
                8, 4 * width, color_space, kCGImageByteOrder32Big | kCGImageAlphaPremultipliedLast
              );

              CGContextDrawImage(
                bctx,
                CGRectMake(0, 0, width, height), cg
              );

              CGImageRelease(cg);
              CGContextRelease(bctx);
            }

            else { // 2: jpg | 3: png | 4: heif | 5: tiff
              NSData *buf = nil;

              if (2 == encoding) {
                napi::object::set(env, ref, "format", napi::string::from(env, "jpg"));
                buf = [ctx JPEGRepresentationOfImage: ci colorSpace: color_space options: @{}];
              }

              else if (3 == encoding) {
                napi::object::set(env, ref, "format", napi::string::from(env, "png"));
                buf = [ctx PNGRepresentationOfImage: ci format: kCIFormatRGBA8 colorSpace: color_space options: @{}];
              }

              else if (4 == encoding) {
                napi::object::set(env, ref, "format", napi::string::from(env, "heif"));
                buf = [ctx HEIFRepresentationOfImage: ci format: kCIFormatRGBA8 colorSpace: color_space options: @{}];
              }

              else if (5 == encoding) {
                napi::object::set(env, ref, "format", napi::string::from(env, "tiff"));
                buf = [ctx TIFFRepresentationOfImage: ci format: kCIFormatRGBA8 colorSpace: color_space options: @{}];
              }

              [buf retain];
              CGColorSpaceRelease(color_space);

              napi::object::set(env, ref, "buffer",
                napi::slice::u8(env, napi::alloc::from(env, buf.length, (void*)buf.bytes, buf,
                [](auto env, auto ptr, auto hint) { [(NSData*)hint release]; }), 0, buf.length)
              );
            }
          }

          return ref;
        }
      })},
    })},
  }));
}

napi::value ModelDescription(napi::env env, NSDictionary<NSString*, MLFeatureDescription*> *descriptions) {
  auto raw = napi::object::empty(env);

  @autoreleasepool {
    for (NSString *key in descriptions) {
      auto ref = napi::object::empty(env);
      napi::object::set(env, raw, key, ref);
      MLFeatureDescription *description = descriptions[key];
      napi::object::set(env, ref, "optional", napi::boolean::from(env, description.optional));

      if ([description type] == MLFeatureTypeInt64) napi::object::set(env, ref, "type", napi::string::from(env, "i64"));
      else if ([description type] == MLFeatureTypeDouble) napi::object::set(env, ref, "type", napi::string::from(env, "f64"));
      else if ([description type] == MLFeatureTypeString) napi::object::set(env, ref, "type", napi::string::from(env, "string"));
      else if ([description type] == MLFeatureTypeInvalid) napi::object::set(env, ref, "type", napi::string::from(env, "invalid"));

      else if ([description type] == MLFeatureTypeDictionary) {
        auto constraint = [description.dictionaryConstraint retain];
        napi::object::set(env, ref, "type", napi::string::from(env, "dict"));
        napi::object::wrap<MLDictionaryConstraint*>(env, ref, constraint, [](auto env, auto ref) { [ref release]; });
        if (constraint.keyType == MLFeatureTypeInt64) napi::object::set(env, ref, "key", napi::string::from(env, "i64"));
        else if (constraint.keyType == MLFeatureTypeString) napi::object::set(env, ref, "key", napi::string::from(env, "string"));
      }

      else if ([description type] == MLFeatureTypeSequence) {
        auto constraint = description.sequenceConstraint;
        napi::object::set(env, ref, "type", napi::string::from(env, "seq"));
        if (MLFeatureTypeInt64 == [constraint.valueDescription type]) napi::object::set(env, ref, "value", napi::string::from(env, "i64"));
        else if (MLFeatureTypeString == [constraint.valueDescription type]) napi::object::set(env, ref, "value", napi::string::from(env, "string"));

        napi::object::set(env, ref, "range", napi::object::from(env, {
          {"length", napi::number::from(env, constraint.countRange.length)},
          {"location", napi::number::from(env, constraint.countRange.location)},
        }));
      }

      else if ([description type] == MLFeatureTypeImage) {
        auto widths = napi::array::empty(env);
        auto heights = napi::array::empty(env);
        napi::object::set(env, ref, "width", widths);
        napi::object::set(env, ref, "height", heights);
        auto constraint = [description.imageConstraint retain];
        napi::object::set(env, ref, "type", napi::string::from(env, "image"));
        napi::object::wrap<MLImageConstraint*>(env, ref, constraint, [](auto env, auto ref) { [ref release]; });

        for (MLImageSize *size in constraint.sizeConstraint.enumeratedImageSizes) {
          napi::array::push(env, widths, napi::number::from(env, size.pixelsWide));
          napi::array::push(env, heights, napi::number::from(env, size.pixelsHigh));
        }
      }

      else if ([description type] == MLFeatureTypeMultiArray) {
        auto shapes = napi::array::empty(env);
        napi::object::set(env, ref, "shape", shapes);
        auto constraint = description.multiArrayConstraint;
        napi::object::set(env, ref, "type", napi::string::from(env, "multiarray"));

        if (constraint.dataType == MLMultiArrayDataTypeInt32) napi::object::set(env, ref, "dtype", napi::string::from(env, "i32"));
        else if (constraint.dataType == MLMultiArrayDataTypeFloat16) napi::object::set(env, ref, "dtype", napi::string::from(env, "f16"));
        else if (constraint.dataType == MLMultiArrayDataTypeFloat32) napi::object::set(env, ref, "dtype", napi::string::from(env, "f32"));
        else if (constraint.dataType == MLMultiArrayDataTypeFloat64) napi::object::set(env, ref, "dtype", napi::string::from(env, "f64"));

        if (constraint.shapeConstraint.type == MLMultiArrayShapeConstraintTypeUnspecified) {
          auto shape = napi::array::empty(env);
          napi::array::push(env, shapes, shape);
          napi::object::set(env, ref, "kind", napi::string::from(env, "unspecified"));
          for (NSNumber *size in constraint.shape) napi::array::push(env, shape, napi::number::from(env, size.unsignedIntValue));
        }

        else if (constraint.shapeConstraint.type == MLMultiArrayShapeConstraintTypeEnumerated) {
          napi::object::set(env, ref, "kind", napi::string::from(env, "enumerated"));

          for (NSArray<NSNumber*> *shape in constraint.shapeConstraint.enumeratedShapes) {
            auto s = napi::array::empty(env); napi::array::push(env, shapes, s);
            for (NSNumber *size in shape) napi::array::push(env, s, napi::number::from(env, size.unsignedIntValue));
          }
        }

        else if (constraint.shapeConstraint.type == MLMultiArrayShapeConstraintTypeRange) {
          napi::object::set(env, ref, "kind", napi::string::from(env, "range"));

          for (NSValue *v in constraint.shapeConstraint.sizeRangeForDimension) {
            auto range = v.rangeValue;

            napi::array::push(env, shapes, napi::object::from(env, {
              {"length", napi::number::from(env, range.length)},
              {"location", napi::number::from(env, range.location)},
            }));
          }
        }
      }
    }
  }

  return raw;
}