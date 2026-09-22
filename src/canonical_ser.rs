// SPDX-License-Identifier: MIT OR Apache-2.0

//! Promote `f32` to `f64` during canonical JSON serialization.
//!
//! serde_json's `Value::Number` formats `f32` differently when the
//! `arbitrary_precision` feature is unified onto this crate: default stores
//! `f as f64`, while `arbitrary_precision` stores the shortest `f32` decimal
//! (`0.1` vs `0.10000000149011612` for `0.1_f32`). Wrapping serialization so
//! every `serialize_f32` becomes `serialize_f64(f64::from(v))` makes the
//! canonical encoder independent of that feature.

use serde::ser::{
    Serialize, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant, Serializer,
};

/// Serialize `value` with every `f32` widened to `f64` before the inner
/// serializer sees it.
pub(crate) fn serialize_f32_as_f64<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    T: Serialize + ?Sized,
    S: Serializer,
{
    value.serialize(F32AsF64(serializer))
}

struct F32AsF64<T>(T);

impl<T: Serialize + ?Sized> Serialize for F32AsF64<&T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_f32_as_f64(self.0, serializer)
    }
}

impl<S: Serializer> Serializer for F32AsF64<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = F32AsF64<S::SerializeSeq>;
    type SerializeTuple = F32AsF64<S::SerializeTuple>;
    type SerializeTupleStruct = F32AsF64<S::SerializeTupleStruct>;
    type SerializeTupleVariant = F32AsF64<S::SerializeTupleVariant>;
    type SerializeMap = F32AsF64<S::SerializeMap>;
    type SerializeStruct = F32AsF64<S::SerializeStruct>;
    type SerializeStructVariant = F32AsF64<S::SerializeStructVariant>;

    fn serialize_bool(self, v: bool) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_bool(v)
    }
    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i8(v)
    }
    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i16(v)
    }
    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i32(v)
    }
    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i64(v)
    }
    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u8(v)
    }
    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u16(v)
    }
    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u32(v)
    }
    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u64(v)
    }
    fn serialize_i128(self, v: i128) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i128(v)
    }
    fn serialize_u128(self, v: u128) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u128(v)
    }
    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_f64(f64::from(v))
    }
    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_f64(v)
    }
    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_char(v)
    }
    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_str(v)
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_bytes(v)
    }
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_none()
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_some(&F32AsF64(value))
    }
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit()
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit_struct(name)
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit_variant(name, variant_index, variant)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_newtype_struct(name, &F32AsF64(value))
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.0
            .serialize_newtype_variant(name, variant_index, variant, &F32AsF64(value))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        self.0.serialize_seq(len).map(F32AsF64)
    }
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.0.serialize_tuple(len).map(F32AsF64)
    }
    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.0.serialize_tuple_struct(name, len).map(F32AsF64)
    }
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.0
            .serialize_tuple_variant(name, variant_index, variant, len)
            .map(F32AsF64)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        self.0.serialize_map(len).map(F32AsF64)
    }
    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.0.serialize_struct(name, len).map(F32AsF64)
    }
    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.0
            .serialize_struct_variant(name, variant_index, variant, len)
            .map(F32AsF64)
    }
    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
}

/// Implement one of serde's sequence-like `Serialize*` traits for
/// `F32AsF64<S>` by forwarding the single element/field method through the
/// `F32AsF64` wrapper and delegating `end()` to the inner accessor. Every impl
/// in this cluster (`SerializeSeq`/`SerializeTuple`/`SerializeTupleStruct`/
/// `SerializeTupleVariant`) has the identical shape, so stamping them out keeps
/// the widening behavior in exactly one place.
macro_rules! impl_forwarding_seq {
    ($trait:ident, $method:ident) => {
        impl<S: $trait> $trait for F32AsF64<S> {
            type Ok = S::Ok;
            type Error = S::Error;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
                self.0.$method(&F32AsF64(value))
            }
            fn end(self) -> Result<Self::Ok, Self::Error> {
                self.0.end()
            }
        }
    };
}

impl_forwarding_seq!(SerializeSeq, serialize_element);
impl_forwarding_seq!(SerializeTuple, serialize_element);
impl_forwarding_seq!(SerializeTupleStruct, serialize_field);
impl_forwarding_seq!(SerializeTupleVariant, serialize_field);

impl<S: SerializeMap> SerializeMap for F32AsF64<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), Self::Error> {
        self.0.serialize_key(&F32AsF64(key))
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_value(&F32AsF64(value))
    }
    fn serialize_entry<K: Serialize + ?Sized, V: Serialize + ?Sized>(
        &mut self,
        key: &K,
        value: &V,
    ) -> Result<(), Self::Error> {
        self.0.serialize_entry(&F32AsF64(key), &F32AsF64(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

/// Implement one of serde's struct-like `Serialize*` traits for
/// `F32AsF64<S>`. `SerializeStruct` and `SerializeStructVariant` share the
/// identical `serialize_field(key, &F32AsF64(value))` + `skip_field` + `end`
/// shape, so the widening wrapper lives in a single spot.
macro_rules! impl_forwarding_struct {
    ($trait:ident) => {
        impl<S: $trait> $trait for F32AsF64<S> {
            type Ok = S::Ok;
            type Error = S::Error;
            fn serialize_field<T: Serialize + ?Sized>(
                &mut self,
                key: &'static str,
                value: &T,
            ) -> Result<(), Self::Error> {
                self.0.serialize_field(key, &F32AsF64(value))
            }
            fn skip_field(&mut self, key: &'static str) -> Result<(), Self::Error> {
                self.0.skip_field(key)
            }
            fn end(self) -> Result<Self::Ok, Self::Error> {
                self.0.end()
            }
        }
    };
}

impl_forwarding_struct!(SerializeStruct);
impl_forwarding_struct!(SerializeStructVariant);

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use serde_json::json;

    fn widen<T: Serialize + ?Sized>(value: &T) -> serde_json::Value {
        serialize_f32_as_f64(value, serde_json::value::Serializer).unwrap()
    }

    macro_rules! ser_once {
        ($ty:ident, |$serializer:ident| $body:expr) => {
            struct $ty;
            impl Serialize for $ty {
                fn serialize<S: Serializer>(&self, $serializer: S) -> Result<S::Ok, S::Error> {
                    $body
                }
            }
        };
    }

    ser_once!(SerBool, |s| s.serialize_bool(true));
    ser_once!(SerI8, |s| s.serialize_i8(-1));
    ser_once!(SerI16, |s| s.serialize_i16(-2));
    ser_once!(SerI32, |s| s.serialize_i32(-3));
    ser_once!(SerI64, |s| s.serialize_i64(-4));
    ser_once!(SerI128, |s| s.serialize_i128(-5));
    ser_once!(SerU8, |s| s.serialize_u8(1));
    ser_once!(SerU16, |s| s.serialize_u16(2));
    ser_once!(SerU32, |s| s.serialize_u32(3));
    ser_once!(SerU64, |s| s.serialize_u64(4));
    ser_once!(SerU128, |s| s.serialize_u128(5));
    ser_once!(SerF32, |s| s.serialize_f32(0.1));
    ser_once!(SerF64, |s| s.serialize_f64(1.5));
    ser_once!(SerChar, |s| s.serialize_char('z'));
    ser_once!(SerStr, |s| s.serialize_str("s"));
    ser_once!(SerBytes, |s| s.serialize_bytes(&[1, 2]));
    ser_once!(SerNone, |s| s.serialize_none());
    ser_once!(SerSomeF32, |s| s.serialize_some(&0.1_f32));
    ser_once!(SerUnit, |s| s.serialize_unit());
    ser_once!(SerUnitStruct, |s| s.serialize_unit_struct("U"));
    ser_once!(SerUnitVariant, |s| s.serialize_unit_variant("E", 0, "Uv"));
    ser_once!(SerNewtypeStruct, |s| s
        .serialize_newtype_struct("N", &0.1_f32));
    ser_once!(SerNewtypeVariant, |s| s
        .serialize_newtype_variant("E", 1, "Nv", &0.1_f32));
    ser_once!(SerSeq, |s| {
        let mut seq = s.serialize_seq(Some(2))?;
        seq.serialize_element(&0.1_f32)?;
        seq.serialize_element(&1_u8)?;
        seq.end()
    });
    ser_once!(SerTuple, |s| {
        let mut tup = s.serialize_tuple(2)?;
        tup.serialize_element(&0.1_f32)?;
        tup.serialize_element(&true)?;
        tup.end()
    });
    ser_once!(SerTupleStruct, |s| {
        let mut tup = s.serialize_tuple_struct("Ts", 1)?;
        tup.serialize_field(&0.1_f32)?;
        tup.end()
    });
    ser_once!(SerTupleVariant, |s| {
        let mut tup = s.serialize_tuple_variant("E", 2, "Tv", 1)?;
        tup.serialize_field(&0.1_f32)?;
        tup.end()
    });
    ser_once!(SerMapEntry, |s| {
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry("k", &0.1_f32)?;
        map.end()
    });
    ser_once!(SerMapKeyValue, |s| {
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_key("k")?;
        map.serialize_value(&0.1_f32)?;
        map.end()
    });
    ser_once!(SerStructSkip, |s| {
        let mut st = s.serialize_struct("S", 2)?;
        st.serialize_field("a", &1_u8)?;
        st.skip_field("b")?;
        st.end()
    });
    ser_once!(SerStructVariant, |s| {
        let mut st = s.serialize_struct_variant("E", 3, "Sv", 2)?;
        st.serialize_field("a", &0.1_f32)?;
        st.skip_field("b")?;
        st.end()
    });
    ser_once!(SerHumanReadable, |s| {
        let human = s.is_human_readable();
        s.serialize_bool(human)
    });

    #[test]
    fn f32_as_f64_adapter_forwards_every_serializer_method() {
        let promoted = json!(0.10000000149011612);
        assert_eq!(widen(&SerBool), json!(true));
        assert_eq!(widen(&SerI8), json!(-1));
        assert_eq!(widen(&SerI16), json!(-2));
        assert_eq!(widen(&SerI32), json!(-3));
        assert_eq!(widen(&SerI64), json!(-4));
        assert_eq!(widen(&SerI128), json!(-5));
        assert_eq!(widen(&SerU8), json!(1));
        assert_eq!(widen(&SerU16), json!(2));
        assert_eq!(widen(&SerU32), json!(3));
        assert_eq!(widen(&SerU64), json!(4));
        assert_eq!(widen(&SerU128), json!(5));
        assert_eq!(widen(&SerF32), promoted);
        assert_eq!(widen(&SerF64), json!(1.5));
        assert_eq!(widen(&SerChar), json!("z"));
        assert_eq!(widen(&SerStr), json!("s"));
        assert_eq!(widen(&SerBytes), json!([1, 2]));
        assert_eq!(widen(&SerNone), json!(null));
        assert_eq!(widen(&SerSomeF32), promoted);
        assert_eq!(widen(&SerUnit), json!(null));
        assert_eq!(widen(&SerUnitStruct), json!(null));
        assert_eq!(widen(&SerUnitVariant), json!("Uv"));
        assert_eq!(widen(&SerNewtypeStruct), promoted);
        assert_eq!(widen(&SerNewtypeVariant), json!({"Nv": promoted}));
        assert_eq!(widen(&SerSeq), json!([promoted, 1]));
        assert_eq!(widen(&SerTuple), json!([promoted, true]));
        assert_eq!(widen(&SerTupleStruct), json!([promoted]));
        assert_eq!(widen(&SerTupleVariant), json!({"Tv": [promoted]}));
        assert_eq!(widen(&SerMapEntry), json!({"k": promoted}));
        assert_eq!(widen(&SerMapKeyValue), json!({"k": promoted}));
        assert_eq!(widen(&SerStructSkip), json!({"a": 1}));
        assert_eq!(widen(&SerStructVariant), json!({"Sv": {"a": promoted}}));
        assert_eq!(widen(&SerHumanReadable), json!(true));
    }
}
