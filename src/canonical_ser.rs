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

impl<S: SerializeSeq> SerializeSeq for F32AsF64<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_element(&F32AsF64(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

impl<S: SerializeTuple> SerializeTuple for F32AsF64<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_element(&F32AsF64(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

impl<S: SerializeTupleStruct> SerializeTupleStruct for F32AsF64<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_field(&F32AsF64(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

impl<S: SerializeTupleVariant> SerializeTupleVariant for F32AsF64<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), Self::Error> {
        self.0.serialize_field(&F32AsF64(value))
    }
    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

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

impl<S: SerializeStruct> SerializeStruct for F32AsF64<S> {
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

impl<S: SerializeStructVariant> SerializeStructVariant for F32AsF64<S> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use serde_json::json;

    fn widen<T: Serialize + ?Sized>(value: &T) -> serde_json::Value {
        serialize_f32_as_f64(value, serde_json::value::Serializer).unwrap()
    }

    /// Calls every `Serializer` / compound-type method on `F32AsF64` so the
    /// forwarding adapter stays covered (canonical messages only hit a subset).
    enum Drive {
        Bool,
        I8,
        I16,
        I32,
        I64,
        I128,
        U8,
        U16,
        U32,
        U64,
        U128,
        F32,
        F64,
        Char,
        Str,
        Bytes,
        None,
        SomeF32,
        Unit,
        UnitStruct,
        UnitVariant,
        NewtypeStruct,
        NewtypeVariant,
        Seq,
        Tuple,
        TupleStruct,
        TupleVariant,
        MapEntry,
        MapKeyValue,
        StructSkip,
        StructVariant,
        HumanReadable,
    }

    impl Serialize for Drive {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            match self {
                Self::Bool => serializer.serialize_bool(true),
                Self::I8 => serializer.serialize_i8(-1),
                Self::I16 => serializer.serialize_i16(-2),
                Self::I32 => serializer.serialize_i32(-3),
                Self::I64 => serializer.serialize_i64(-4),
                Self::I128 => serializer.serialize_i128(-5),
                Self::U8 => serializer.serialize_u8(1),
                Self::U16 => serializer.serialize_u16(2),
                Self::U32 => serializer.serialize_u32(3),
                Self::U64 => serializer.serialize_u64(4),
                Self::U128 => serializer.serialize_u128(5),
                Self::F32 => serializer.serialize_f32(0.1),
                Self::F64 => serializer.serialize_f64(1.5),
                Self::Char => serializer.serialize_char('z'),
                Self::Str => serializer.serialize_str("s"),
                Self::Bytes => serializer.serialize_bytes(&[1, 2]),
                Self::None => serializer.serialize_none(),
                Self::SomeF32 => serializer.serialize_some(&0.1_f32),
                Self::Unit => serializer.serialize_unit(),
                Self::UnitStruct => serializer.serialize_unit_struct("U"),
                Self::UnitVariant => serializer.serialize_unit_variant("E", 0, "Uv"),
                Self::NewtypeStruct => serializer.serialize_newtype_struct("N", &0.1_f32),
                Self::NewtypeVariant => {
                    serializer.serialize_newtype_variant("E", 1, "Nv", &0.1_f32)
                }
                Self::Seq => {
                    let mut seq = serializer.serialize_seq(Some(2))?;
                    seq.serialize_element(&0.1_f32)?;
                    seq.serialize_element(&1_u8)?;
                    seq.end()
                }
                Self::Tuple => {
                    let mut tup = serializer.serialize_tuple(2)?;
                    tup.serialize_element(&0.1_f32)?;
                    tup.serialize_element(&true)?;
                    tup.end()
                }
                Self::TupleStruct => {
                    let mut tup = serializer.serialize_tuple_struct("Ts", 1)?;
                    tup.serialize_field(&0.1_f32)?;
                    tup.end()
                }
                Self::TupleVariant => {
                    let mut tup = serializer.serialize_tuple_variant("E", 2, "Tv", 1)?;
                    tup.serialize_field(&0.1_f32)?;
                    tup.end()
                }
                Self::MapEntry => {
                    let mut map = serializer.serialize_map(Some(1))?;
                    map.serialize_entry("k", &0.1_f32)?;
                    map.end()
                }
                Self::MapKeyValue => {
                    let mut map = serializer.serialize_map(Some(1))?;
                    map.serialize_key("k")?;
                    map.serialize_value(&0.1_f32)?;
                    map.end()
                }
                Self::StructSkip => {
                    let mut st = serializer.serialize_struct("S", 2)?;
                    st.serialize_field("a", &1_u8)?;
                    st.skip_field("b")?;
                    st.end()
                }
                Self::StructVariant => {
                    let mut st = serializer.serialize_struct_variant("E", 3, "Sv", 2)?;
                    st.serialize_field("a", &0.1_f32)?;
                    st.skip_field("b")?;
                    st.end()
                }
                Self::HumanReadable => {
                    let human = serializer.is_human_readable();
                    serializer.serialize_bool(human)
                }
            }
        }
    }

    #[test]
    fn f32_as_f64_adapter_forwards_every_serializer_method() {
        let promoted = json!(0.10000000149011612);
        assert_eq!(widen(&Drive::Bool), json!(true));
        assert_eq!(widen(&Drive::I8), json!(-1));
        assert_eq!(widen(&Drive::I16), json!(-2));
        assert_eq!(widen(&Drive::I32), json!(-3));
        assert_eq!(widen(&Drive::I64), json!(-4));
        assert_eq!(widen(&Drive::I128), json!(-5));
        assert_eq!(widen(&Drive::U8), json!(1));
        assert_eq!(widen(&Drive::U16), json!(2));
        assert_eq!(widen(&Drive::U32), json!(3));
        assert_eq!(widen(&Drive::U64), json!(4));
        assert_eq!(widen(&Drive::U128), json!(5));
        assert_eq!(widen(&Drive::F32), promoted);
        assert_eq!(widen(&Drive::F64), json!(1.5));
        assert_eq!(widen(&Drive::Char), json!("z"));
        assert_eq!(widen(&Drive::Str), json!("s"));
        assert_eq!(widen(&Drive::Bytes), json!([1, 2]));
        assert_eq!(widen(&Drive::None), json!(null));
        assert_eq!(widen(&Drive::SomeF32), promoted);
        assert_eq!(widen(&Drive::Unit), json!(null));
        assert_eq!(widen(&Drive::UnitStruct), json!(null));
        assert_eq!(widen(&Drive::UnitVariant), json!("Uv"));
        assert_eq!(widen(&Drive::NewtypeStruct), promoted);
        assert_eq!(widen(&Drive::NewtypeVariant), json!({"Nv": promoted}));
        assert_eq!(widen(&Drive::Seq), json!([promoted, 1]));
        assert_eq!(widen(&Drive::Tuple), json!([promoted, true]));
        assert_eq!(widen(&Drive::TupleStruct), json!([promoted]));
        assert_eq!(widen(&Drive::TupleVariant), json!({"Tv": [promoted]}));
        assert_eq!(widen(&Drive::MapEntry), json!({"k": promoted}));
        assert_eq!(widen(&Drive::MapKeyValue), json!({"k": promoted}));
        assert_eq!(widen(&Drive::StructSkip), json!({"a": 1}));
        assert_eq!(widen(&Drive::StructVariant), json!({"Sv": {"a": promoted}}));
        assert_eq!(widen(&Drive::HumanReadable), json!(true));
    }
}
