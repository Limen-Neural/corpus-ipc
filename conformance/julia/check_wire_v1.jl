#!/usr/bin/env julia
# SPDX-License-Identifier: MIT OR Apache-2.0
# Offline wire-v1 conformance harness (LIM-1327).
#
# Same hashed corpus and outcomes as conformance/python/check_wire_v1.py and
# the Rust tests in tests/wire_v1_vectors.rs. JSON.jl is the only dependency
# (see Project.toml / Manifest.toml). SHA is a Julia stdlib.
#
#   julia --project=conformance/julia -e 'using Pkg; Pkg.instantiate()'
#   julia --project=conformance/julia conformance/julia/check_wire_v1.jl

try
    using JSON
    using SHA
catch
    println(
        stderr,
        "missing JSON.jl; from the repo root run: julia --project=conformance/julia -e 'using Pkg; Pkg.instantiate()'",
    )
    exit(1)
end

const CURRENT = 1
const MIN_SUPPORTED = 1
const LEGACY_UNVERSIONED = 1

const MAX_CHANNEL_VALUES = 65_536
const MAX_SPIKE_EVENTS = 65_536
const MAX_TRACES = 65_536
const MAX_GRADIENTS = 65_536
const MAX_METADATA_ENTRIES = 256
const MAX_STRING_BYTES = 1_024
const MAX_AGGREGATE = 131_072

const UNIT_VARIANTS = ("Ping", "Shutdown", "TrainingComplete")
const KNOWN_VARIANTS = (
    "Spikes",
    "Embeddings",
    "Stimuli",
    "Neuromodulators",
    "Loss",
    "ConfigUpdate",
    "GradientUpdate",
    "EligibilityTraces",
    UNIT_VARIANTS...,
)
const EXPECTATIONS = ("decode_ok", "too_old", "too_new", "payload_error", "json_error")
const DECODE_OK_KINDS = ("positive", "legacy", "alias", "normalization")

struct WireError <: Exception
    kind::String
    reason::String
end

Base.showerror(io::IO, err::WireError) = print(io, err.kind, ": ", err.reason)

payload_error(reason::AbstractString) = WireError("payload_error", String(reason))

function classify_wire_version(version::Integer)::String
    if version < MIN_SUPPORTED
        return "too_old"
    elseif version > CURRENT
        return "too_new"
    end
    return "supported"
end

function fail(name::AbstractString, detail::AbstractString)::Union{}
    println(stderr, "FAIL ", name, ": ", detail)
    exit(1)
end

# --- f32 / canonical JSON numbers ------------------------------------------

function uint_to_f32(n::Integer)::Float64
    n < 0 && error("uint_to_f32 expects a non-negative integer")
    n == 0 && return 0.0
    exp = ndigits(BigInt(n); base=2) - 1
    exp >= 128 && return Inf
    if exp <= 23
        return Float64(Float32(n))
    end
    shift = exp - 23
    bn = BigInt(n)
    significand = bn >> shift
    remainder = bn & ((BigInt(1) << shift) - 1)
    halfway = BigInt(1) << (shift - 1)
    if remainder > halfway || (remainder == halfway && isodd(significand))
        significand += 1
        if significand == (BigInt(1) << 24)
            significand >>= 1
            exp += 1
            exp >= 128 && return Inf
        end
    end
    frac = significand & ((BigInt(1) << 23) - 1)
    raw = UInt32((exp + 127) << 23) | UInt32(frac)
    return Float64(reinterpret(Float32, raw))
end

function int_to_f32(n::Integer)::Float64
    if n < 0
        rounded = uint_to_f32(-n)
        return rounded == 0.0 ? -0.0 : -rounded
    end
    return uint_to_f32(n)
end

function as_f32(value, path::AbstractString)::Float64
    if value isa Bool || !(value isa Real)
        throw(payload_error("$path must be a finite JSON number"))
    end
    if value isa Integer
        if value < -(BigInt(2)^63) || value > (BigInt(2)^64 - 1)
            throw(payload_error("$path integer does not fit the wire numeric range"))
        end
        rounded = int_to_f32(value)
    else
        x = Float64(value)
        isfinite(x) || throw(payload_error("$path is non-finite"))
        rounded = Float64(Float32(x))
    end
    isfinite(rounded) || throw(payload_error("$path is non-finite"))
    return rounded
end

function parse_shortest(token::AbstractString)::Tuple{String,Int}
    text = lowercase(String(token))
    if occursin('e', text)
        mantissa, exp_text = split(text, 'e')
        exp = parse(Int, exp_text)
    else
        mantissa = text
        exp = 0
    end
    if occursin('.', mantissa)
        whole, frac = split(mantissa, '.')
    else
        whole = mantissa
        frac = ""
    end
    raw = collect(whole * frac)
    lead = 0
    while lead < length(raw) - 1 && raw[lead + 1] == '0'
        lead += 1
    end
    digits = String(raw[(lead + 1):end])
    dec_exp = exp + length(collect(whole)) - 1 - lead
    return digits, dec_exp
end

function render_zmij(digits::AbstractString, dec_exp::Int)::String
    chars = collect(String(digits))
    length_digits = length(chars)
    if -5 <= dec_exp <= 15
        if length_digits - 1 <= dec_exp
            zeros = dec_exp - (length_digits - 1)
            return string(digits, "0"^zeros, ".0")
        elseif dec_exp >= 0
            pos = dec_exp + 1
            return string(String(chars[1:pos]), ".", String(chars[(pos + 1):end]))
        else
            zeros = -dec_exp - 1
            return string("0.", "0"^zeros, digits)
        end
    end
    mantissa = length_digits == 1 ? String(digits) : string(chars[1], ".", String(chars[2:end]))
    sign = dec_exp >= 0 ? "+" : "-"
    # serde_json / zmij writes the exponent without a leading zero: e-7, e+16.
    return string(mantissa, "e", sign, abs(dec_exp))
end

function format_f64(value::Float64)::String
    isfinite(value) || throw(payload_error("non-finite float cannot be encoded"))
    if value == 0.0
        return signbit(value) ? "-0.0" : "0.0"
    end
    negative = signbit(value)
    # hash=false omits a trailing ".0" so the significand matches zmij's digits.
    # Notation (fixed vs scientific, exponent sign) is applied afterwards.
    digits, dec_exp = parse_shortest(Base.Ryu.writeshortest(abs(value), false, false, false))
    body = render_zmij(digits, dec_exp)
    return negative ? string("-", body) : body
end

function dumps(value)::String
    if value === nothing
        return "null"
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        return format_f64(Float64(value))
    elseif value isa AbstractString
        return JSON.json(String(value))
    elseif value isa AbstractVector
        return string("[", join((dumps(item) for item in value), ","), "]")
    elseif value isa AbstractDict
        keys_sorted = sort!(collect(keys(value)); by=key -> Vector{UInt8}(codeunits(String(key))))
        parts = String[string(dumps(String(key)), ":", dumps(value[key])) for key in keys_sorted]
        return string("{", join(parts, ","), "}")
    end
    error("cannot encode $(typeof(value))")
end

# --- model -----------------------------------------------------------------

struct SpikeEvent
    channel::Int
    time::Int
    strength::Float64
end

struct BatchMetadata
    processing_latency_ns::Union{BigInt,Nothing}
    source::Union{String,Nothing}
    custom::Dict{String,String}
end

struct SpikeBatch
    session_id::Union{String,Nothing}
    batch_id::BigInt
    timestamp::BigInt
    spikes::Vector{SpikeEvent}
    metadata::Union{BatchMetadata,Nothing}
end

struct EmbeddingBatch
    session_id::Union{String,Nothing}
    batch_id::BigInt
    embedding::Vector{Float64}
    sequence_length::BigInt
end

struct StimulusBatch
    session_id::Union{String,Nothing}
    batch_id::BigInt
    timestamp::BigInt
    values::Vector{Float64}
    valid_mask::Union{Vector{Bool},Nothing}
    metadata::Union{BatchMetadata,Nothing}
end

struct NeuromodulatorSnapshot
    tick::BigInt
    dopamine::Float64
    cortisol::Float64
    acetylcholine::Float64
    tempo::Float64
end

struct ConfigValue
    kind::Symbol
    value
end

struct ConfigPayload
    session_id::Union{String,Nothing}
    config::Dict{String,ConfigValue}
end

struct GradientUpdateRow
    layer_id::String
    gradients::Vector{Float64}
    eligibility_trace::Union{Vector{Float64},Nothing}
end

struct GradientBatch
    session_id::String
    batch_id::BigInt
    gradients::Vector{GradientUpdateRow}
end

struct TraceData
    channel_id::Int
    trace_value::Float64
    last_spike_time::Int
end

struct TraceBatch
    session_id::String
    batch_id::BigInt
    traces::Vector{TraceData}
end

struct Message
    variant::String
    body
end

# --- validation helpers ----------------------------------------------------

function string_bytes(value::AbstractString, path::AbstractString)
    sizeof(value) > MAX_STRING_BYTES &&
        throw(payload_error("$path exceeds $MAX_STRING_BYTES bytes"))
    return nothing
end

function check_count(actual::Integer, limit::Integer, path::AbstractString)
    actual > limit && throw(payload_error("$path length $actual exceeds $limit"))
    return nothing
end

function check_finite(value::Float64, path::AbstractString)
    isfinite(value) || throw(payload_error("$path is non-finite"))
    return nothing
end

function check_range(value::Float64, low::Float64, high::Float64, path::AbstractString)
    check_finite(value, path)
    if value < low || value > high
        throw(payload_error("$path $(format_f64(value)) is outside [$low, $high]"))
    end
    return nothing
end

function add_total(total::Int, count::Int, path::AbstractString)::Int
    total += count
    total > MAX_AGGREGATE && throw(payload_error("$path aggregate exceeds $MAX_AGGREGATE"))
    return total
end

function require_object(value, path::AbstractString)::AbstractDict
    value isa AbstractDict || throw(payload_error("$path must be a JSON object"))
    return value
end

function require_key(obj::AbstractDict, key::AbstractString, path::AbstractString)
    haskey(obj, key) || throw(payload_error("missing $path"))
    return obj[key]
end

function as_u64(value, path::AbstractString)::BigInt
    if value isa Bool || !(value isa Integer)
        throw(payload_error("$path must be a non-negative integer"))
    end
    if value < 0 || value > typemax(UInt64)
        throw(payload_error("$path is outside u64"))
    end
    return BigInt(value)
end

function as_u32(value, path::AbstractString)::Int
    number = as_u64(value, path)
    number > typemax(UInt32) && throw(payload_error("$path is outside u32"))
    return Int(number)
end

function as_u16(value, path::AbstractString)::Int
    number = as_u64(value, path)
    number > typemax(UInt16) && throw(payload_error("$path is outside u16"))
    return Int(number)
end

function as_i64(value, path::AbstractString)::BigInt
    if value isa Bool || !(value isa Integer)
        throw(payload_error("$path must be an integer"))
    end
    if value < typemin(Int64) || value > typemax(Int64)
        throw(payload_error("$path is outside i64"))
    end
    return BigInt(value)
end

function opt_session(value, path::AbstractString)::Union{String,Nothing}
    value === nothing && return nothing
    value isa AbstractString || throw(payload_error("$path must be a string or null"))
    string_bytes(value, path)
    return String(value)
end

function required_session(value, path::AbstractString)::String
    value isa AbstractString || throw(payload_error("$path must be a string"))
    string_bytes(value, path)
    return String(value)
end

function f32_list(value, path::AbstractString, limit::Int)::Vector{Float64}
    value isa AbstractVector || throw(payload_error("$path must be an array"))
    check_count(length(value), limit, path)
    return [as_f32(item, "$path[$(index - 1)]") for (index, item) in enumerate(value)]
end

function optional_metadata(obj)
    # Option fields without deserialize_with accept a missing key as None.
    haskey(obj, "metadata") || return nothing
    metadata = obj["metadata"]
    metadata === nothing && return nothing
    return decode_metadata(metadata, "metadata")
end

function decode_metadata(value, path::AbstractString)::BatchMetadata
    obj = require_object(value, path)
    latency = nothing
    if haskey(obj, "processing_latency_ns") && obj["processing_latency_ns"] !== nothing
        latency = as_u64(obj["processing_latency_ns"], "$path.processing_latency_ns")
    end
    source = haskey(obj, "source") ? opt_session(obj["source"], "$path.source") : nothing
    custom = Dict{String,String}()
    if haskey(obj, "custom")
        custom_raw = obj["custom"]
        custom_raw isa AbstractDict || throw(payload_error("$path.custom must be an object"))
        check_count(length(custom_raw), MAX_METADATA_ENTRIES, "$path.custom")
        for (key, item) in custom_raw
            key == "" && throw(payload_error("$path.custom.<empty> is not a valid key"))
            string_bytes(key, "$path.custom.<key>")
            item isa AbstractString || throw(payload_error("$path.custom.$key must be a string"))
            string_bytes(item, "$path.custom.<value>")
            custom[String(key)] = String(item)
        end
    end
    check_count(length(custom), MAX_METADATA_ENTRIES, "$path.custom")
    return BatchMetadata(latency, source, custom)
end

function validate_metadata(metadata::Union{BatchMetadata,Nothing}, already::Int, path::AbstractString)
    total = add_total(0, already, path)
    metadata === nothing && return nothing
    metadata.source === nothing || string_bytes(metadata.source, "$path.source")
    check_count(length(metadata.custom), MAX_METADATA_ENTRIES, "$path.custom")
    for (key, item) in metadata.custom
        key == "" && throw(payload_error("$path.custom.<empty> is not a valid key"))
        string_bytes(key, "$path.custom.<key>")
        string_bytes(item, "$path.custom.<value>")
    end
    add_total(total, length(metadata.custom), path)
    return nothing
end

function decode_spike(value, path::AbstractString)::SpikeEvent
    obj = require_object(value, path)
    event = SpikeEvent(
        as_u16(require_key(obj, "channel", "$path.channel"), "$path.channel"),
        as_u32(require_key(obj, "time", "$path.time"), "$path.time"),
        as_f32(require_key(obj, "strength", "$path.strength"), "$path.strength"),
    )
    check_finite(event.strength, "$path.strength")
    return event
end

function decode_spikes(value)::SpikeBatch
    obj = require_object(value, "Spikes")
    session = haskey(obj, "session_id") ? opt_session(obj["session_id"], "session_id") : nothing
    spikes_raw = require_key(obj, "spikes", "spikes")
    spikes_raw isa AbstractVector || throw(payload_error("spikes must be an array"))
    check_count(length(spikes_raw), MAX_SPIKE_EVENTS, "spikes")
    spikes = [decode_spike(item, "spikes[$(index - 1)]") for (index, item) in enumerate(spikes_raw)]
    decoded_meta = optional_metadata(obj)
    batch = SpikeBatch(
        session,
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        as_u64(require_key(obj, "timestamp", "timestamp"), "timestamp"),
        spikes,
        decoded_meta,
    )
    batch.session_id === nothing || string_bytes(batch.session_id, "session_id")
    validate_metadata(batch.metadata, length(batch.spikes), "spikes")
    return batch
end

function decode_embeddings(value)::EmbeddingBatch
    obj = require_object(value, "Embeddings")
    embedding = f32_list(require_key(obj, "embedding", "embedding"), "embedding", MAX_CHANNEL_VALUES)
    sequence = as_u64(require_key(obj, "sequence_length", "sequence_length"), "sequence_length")
    check_count(sequence, MAX_AGGREGATE, "sequence_length")
    for (index, item) in enumerate(embedding)
        check_finite(item, "embedding[$(index - 1)]")
    end
    add_total(0, length(embedding), "embedding")
    return EmbeddingBatch(
        opt_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        embedding,
        sequence,
    )
end

function decode_stimuli(value)::StimulusBatch
    obj = require_object(value, "Stimuli")
    values = f32_list(require_key(obj, "values", "values"), "values", MAX_CHANNEL_VALUES)
    for (index, item) in enumerate(values)
        check_finite(item, "values[$(index - 1)]")
    end
    mask_raw = require_key(obj, "valid_mask", "valid_mask")
    mask::Union{Vector{Bool},Nothing} = nothing
    if mask_raw !== nothing
        mask_raw isa AbstractVector || throw(payload_error("valid_mask must be an array or null"))
        check_count(length(mask_raw), MAX_CHANNEL_VALUES, "valid_mask")
        parsed = Bool[]
        for (index, item) in enumerate(mask_raw)
            item isa Bool || throw(payload_error("valid_mask[$(index - 1)] must be a boolean"))
            push!(parsed, item)
        end
        if length(parsed) != length(values)
            throw(
                payload_error(
                    "valid_mask length $(length(parsed)) does not match values length $(length(values))",
                ),
            )
        end
        mask = parsed
    end
    decoded_meta = optional_metadata(obj)
    validate_metadata(decoded_meta, length(values), "values")
    return StimulusBatch(
        opt_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        as_u64(require_key(obj, "timestamp", "timestamp"), "timestamp"),
        values,
        mask,
        decoded_meta,
    )
end

function decode_neuromodulators(value)::NeuromodulatorSnapshot
    obj = require_object(value, "Neuromodulators")
    snapshot = NeuromodulatorSnapshot(
        as_i64(require_key(obj, "tick", "tick"), "tick"),
        as_f32(require_key(obj, "dopamine", "dopamine"), "dopamine"),
        as_f32(require_key(obj, "cortisol", "cortisol"), "cortisol"),
        as_f32(require_key(obj, "acetylcholine", "acetylcholine"), "acetylcholine"),
        as_f32(require_key(obj, "tempo", "tempo"), "tempo"),
    )
    check_range(snapshot.dopamine, 0.0, 1.0, "dopamine")
    check_range(snapshot.cortisol, 0.0, 1.0, "cortisol")
    check_range(snapshot.acetylcholine, 0.0, 1.0, "acetylcholine")
    check_range(snapshot.tempo, 0.5, 2.0, "tempo")
    return snapshot
end

function decode_loss(value)::Float64
    loss = as_f32(value, "Loss")
    check_finite(loss, "Loss")
    return loss
end

function decode_config_value(value, path::AbstractString)::ConfigValue
    if value isa Bool
        return ConfigValue(:bool, value)
    elseif value isa AbstractString
        string_bytes(value, path)
        return ConfigValue(:string, String(value))
    elseif value isa AbstractVector
        numbers = f32_list(value, path, MAX_CHANNEL_VALUES)
        for (index, item) in enumerate(numbers)
            check_finite(item, "$path[$(index - 1)]")
        end
        return ConfigValue(:array, numbers)
    elseif value isa AbstractDict
        throw(payload_error("$path must be a number, string, boolean, or array of floats"))
    end
    number = as_f32(value, path)
    check_finite(number, path)
    return ConfigValue(:float, number)
end

function decode_config(value)::ConfigPayload
    obj = require_object(value, "ConfigUpdate")
    raw = require_key(obj, "config", "config")
    raw isa AbstractDict || throw(payload_error("config must be an object"))
    check_count(length(raw), MAX_METADATA_ENTRIES, "config")
    config = Dict{String,ConfigValue}()
    total = add_total(0, length(raw), "config")
    for (key, item) in raw
        key == "" && throw(payload_error("config.<empty> is not a valid key"))
        string_bytes(key, "config.<key>")
        decoded = decode_config_value(item, "config.$key")
        config[String(key)] = decoded
        if decoded.kind == :array
            total = add_total(total, length(decoded.value), "config")
        end
    end
    return ConfigPayload(
        opt_session(require_key(obj, "session_id", "session_id"), "session_id"),
        config,
    )
end

function decode_gradient_row(value, path::AbstractString)::GradientUpdateRow
    obj = require_object(value, path)
    gradients = f32_list(
        require_key(obj, "gradients", "$path.gradients"),
        "$path.gradients",
        MAX_CHANNEL_VALUES,
    )
    for (index, item) in enumerate(gradients)
        check_finite(item, "$path.gradients[$(index - 1)]")
    end
    trace = nothing
    if haskey(obj, "eligibility_trace") && obj["eligibility_trace"] !== nothing
        trace = f32_list(
            obj["eligibility_trace"],
            "$path.eligibility_trace",
            MAX_CHANNEL_VALUES,
        )
        for (index, item) in enumerate(trace)
            check_finite(item, "$path.eligibility_trace[$(index - 1)]")
        end
    end
    layer = required_session(require_key(obj, "layer_id", "$path.layer_id"), "$path.layer_id")
    return GradientUpdateRow(layer, gradients, trace)
end

function decode_gradients(value)::GradientBatch
    obj = require_object(value, "GradientUpdate")
    raw = require_key(obj, "gradients", "gradients")
    raw isa AbstractVector || throw(payload_error("gradients must be an array"))
    check_count(length(raw), MAX_GRADIENTS, "gradients")
    rows = [
        decode_gradient_row(item, "gradients[$(index - 1)]") for (index, item) in enumerate(raw)
    ]
    seen = Set{String}()
    total = add_total(0, length(rows), "gradients")
    for row in rows
        row.layer_id in seen && throw(payload_error("duplicate layer_id $(repr(row.layer_id))"))
        push!(seen, row.layer_id)
        total = add_total(total, length(row.gradients), "gradients")
        if row.eligibility_trace !== nothing
            total = add_total(total, length(row.eligibility_trace), "gradients")
        end
    end
    return GradientBatch(
        required_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        rows,
    )
end

function decode_trace(value, path::AbstractString)::TraceData
    obj = require_object(value, path)
    has_channel = haskey(obj, "channel_id")
    has_alias = haskey(obj, "neuron_id")
    if has_channel && has_alias
        throw(payload_error("$path has both channel_id and neuron_id"))
    elseif has_channel
        channel = as_u16(obj["channel_id"], "$path.channel_id")
    elseif has_alias
        channel = as_u16(obj["neuron_id"], "$path.neuron_id")
    else
        throw(payload_error("missing $path.channel_id"))
    end
    data = TraceData(
        channel,
        as_f32(require_key(obj, "trace_value", "$path.trace_value"), "$path.trace_value"),
        as_u32(
            require_key(obj, "last_spike_time", "$path.last_spike_time"),
            "$path.last_spike_time",
        ),
    )
    check_finite(data.trace_value, "$path.trace_value")
    return data
end

function decode_traces(value)::TraceBatch
    obj = require_object(value, "EligibilityTraces")
    raw = require_key(obj, "traces", "traces")
    raw isa AbstractVector || throw(payload_error("traces must be an array"))
    check_count(length(raw), MAX_TRACES, "traces")
    traces = [decode_trace(item, "traces[$(index - 1)]") for (index, item) in enumerate(raw)]
    seen = Set{Int}()
    for trace in traces
        trace.channel_id in seen &&
            throw(payload_error("duplicate channel_id $(trace.channel_id)"))
        push!(seen, trace.channel_id)
    end
    add_total(0, length(traces), "traces")
    return TraceBatch(
        required_session(require_key(obj, "session_id", "session_id"), "session_id"),
        as_u64(require_key(obj, "batch_id", "batch_id"), "batch_id"),
        traces,
    )
end

const DECODERS = Dict{String,Function}(
    "Spikes" => decode_spikes,
    "Embeddings" => decode_embeddings,
    "Stimuli" => decode_stimuli,
    "Neuromodulators" => decode_neuromodulators,
    "Loss" => decode_loss,
    "ConfigUpdate" => decode_config,
    "GradientUpdate" => decode_gradients,
    "EligibilityTraces" => decode_traces,
)

function decode_payload(value)::Message
    if value isa AbstractString
        text = String(value)
        text in UNIT_VARIANTS && return Message(text, nothing)
        throw(payload_error("unknown IpcMessage variant $(repr(text))"))
    end
    if !(value isa AbstractDict)
        throw(
            WireError(
                "not_an_object",
                "IpcMessage must be an externally tagged object or a unit-variant string",
            ),
        )
    end
    if haskey(value, "type") && Set(keys(value)) != Set(["type"])
        throw(
            payload_error(
                "internally tagged message is not supported (found type=$(repr(value["type"]))); expected an externally tagged IpcMessage",
            ),
        )
    end
    if length(value) != 1
        found = sort!(collect(String.(keys(value))))
        throw(
            payload_error(
                "externally tagged IpcMessage must have exactly one variant key, found $found",
            ),
        )
    end
    name = String(first(keys(value)))
    body = value[name]
    if name in UNIT_VARIANTS
        if !(body === nothing || body == Dict())
            throw(payload_error("$name is a unit variant and must be a JSON string"))
        end
        return Message(name, nothing)
    end
    haskey(DECODERS, name) || throw(payload_error("unknown IpcMessage variant $(repr(name))"))
    return Message(name, DECODERS[name](body))
end

function parse_version(value)::Int
    if value isa Bool || !(value isa Integer) || value < 0 || value > typemax(UInt32)
        throw(
            WireError(
                "invalid_version",
                "wire_version must be a non-negative integer fitting u32, got $(repr(value))",
            ),
        )
    end
    return Int(value)
end

# `String(::Vector{UInt8})` takes ownership and truncates the vector.
function bytes_string(data::Vector{UInt8})::String
    text = String(copy(data))
    isvalid(text) || throw(WireError("json_error", "envelope JSON error: invalid utf-8"))
    return text
end

function decode_text(data::Vector{UInt8})::String
    return bytes_string(data)
end

function decode_message(data::Vector{UInt8})::Message
    text = decode_text(data)
    value = try
        JSON.parse(text; inttype=BigInt)
    catch err
        err isa WireError && rethrow()
        err isa InterruptException && rethrow()
        throw(WireError("json_error", "envelope JSON error: " * sprint(showerror, err)))
    end
    if value isa AbstractDict && haskey(value, "wire_version")
        version = parse_version(value["wire_version"])
        kind = classify_wire_version(version)
        if kind == "too_old"
            throw(
                WireError(
                    "too_old",
                    "wire version $version is too old (minimum supported is $MIN_SUPPORTED); payload was not interpreted",
                ),
            )
        elseif kind == "too_new"
            throw(
                WireError(
                    "too_new",
                    "wire version $version is too new (current is $CURRENT); payload was not interpreted",
                ),
            )
        end
        haskey(value, "payload") ||
            throw(WireError("missing_payload", "missing payload on compatibility envelope"))
        return decode_payload(value["payload"])
    end
    if classify_wire_version(LEGACY_UNVERSIONED) != "supported"
        throw(WireError("too_old", "legacy unversioned JSON is outside the supported window"))
    end
    return decode_payload(value)
end

# --- canonical encode ------------------------------------------------------

function encode_metadata(metadata::Union{BatchMetadata,Nothing})
    metadata === nothing && return nothing
    return Dict{String,Any}(
        "processing_latency_ns" => metadata.processing_latency_ns,
        "source" => metadata.source,
        "custom" => metadata.custom,
    )
end

function encode_config_value(value::ConfigValue)
    value.kind in (:float, :array, :bool, :string) && return value.value
    throw(payload_error("cannot encode config value $(value.kind)"))
end

function encode_payload(message::Message)
    message.variant in UNIT_VARIANTS && return message.variant
    body = message.body
    if message.variant == "Loss"
        return Dict("Loss" => body)
    elseif message.variant == "Spikes"
        return Dict(
            "Spikes" => Dict{String,Any}(
                "session_id" => body.session_id,
                "batch_id" => body.batch_id,
                "timestamp" => body.timestamp,
                "spikes" => [
                    Dict{String,Any}(
                        "channel" => spike.channel,
                        "time" => spike.time,
                        "strength" => spike.strength,
                    ) for spike in body.spikes
                ],
                "metadata" => encode_metadata(body.metadata),
            ),
        )
    elseif message.variant == "Embeddings"
        return Dict(
            "Embeddings" => Dict{String,Any}(
                "session_id" => body.session_id,
                "batch_id" => body.batch_id,
                "embedding" => body.embedding,
                "sequence_length" => body.sequence_length,
            ),
        )
    elseif message.variant == "Stimuli"
        return Dict(
            "Stimuli" => Dict{String,Any}(
                "session_id" => body.session_id,
                "batch_id" => body.batch_id,
                "timestamp" => body.timestamp,
                "values" => body.values,
                "valid_mask" => body.valid_mask,
                "metadata" => encode_metadata(body.metadata),
            ),
        )
    elseif message.variant == "Neuromodulators"
        return Dict(
            "Neuromodulators" => Dict{String,Any}(
                "tick" => body.tick,
                "dopamine" => body.dopamine,
                "cortisol" => body.cortisol,
                "acetylcholine" => body.acetylcholine,
                "tempo" => body.tempo,
            ),
        )
    elseif message.variant == "ConfigUpdate"
        return Dict(
            "ConfigUpdate" => Dict{String,Any}(
                "session_id" => body.session_id,
                "config" => Dict(
                    key => encode_config_value(item) for (key, item) in body.config
                ),
            ),
        )
    elseif message.variant == "GradientUpdate"
        return Dict(
            "GradientUpdate" => Dict{String,Any}(
                "session_id" => body.session_id,
                "batch_id" => body.batch_id,
                "gradients" => [
                    Dict{String,Any}(
                        "layer_id" => row.layer_id,
                        "gradients" => row.gradients,
                        "eligibility_trace" => row.eligibility_trace,
                    ) for row in body.gradients
                ],
            ),
        )
    elseif message.variant == "EligibilityTraces"
        return Dict(
            "EligibilityTraces" => Dict{String,Any}(
                "session_id" => body.session_id,
                "batch_id" => body.batch_id,
                "traces" => [
                    Dict{String,Any}(
                        "channel_id" => trace.channel_id,
                        "trace_value" => trace.trace_value,
                        "last_spike_time" => trace.last_spike_time,
                    ) for trace in body.traces
                ],
            ),
        )
    end
    throw(payload_error("cannot encode variant $(message.variant)"))
end

function encode_canonical(message::Message)::Vector{UInt8}
    text = dumps(Dict("payload" => encode_payload(message), "wire_version" => CURRENT))
    return Vector{UInt8}(codeunits(text))
end

# --- corpus checks ---------------------------------------------------------

vector_dir() = normpath(joinpath(@__DIR__, "..", "..", "test-vectors", "wire-v1"))

function load_manifest(root::AbstractString)
    path = joinpath(root, "manifest.json")
    manifest = try
        JSON.parse(read(path, String); inttype=BigInt)
    catch err
        println(stderr, "cannot read manifest: ", sprint(showerror, err))
        exit(1)
    end
    manifest["schema_version"] == 1 || (println(stderr, "manifest schema_version must be 1"); exit(1))
    occursin("wire v1", String(manifest["encoding_profile"])) ||
        (println(stderr, "manifest encoding_profile must name wire v1"); exit(1))
    return manifest
end

function check_manifest_hashes(root::AbstractString, manifest)
    fixtures = manifest["fixtures"]
    seen = Set{String}()
    for entry in fixtures
        name = String(entry["name"])
        name in seen && (println(stderr, "duplicate fixture name $name"); exit(1))
        push!(seen, name)
        expect = String(entry["expect"])
        expect in EXPECTATIONS || (println(stderr, "$name: unknown expectation $expect"); exit(1))
        if entry["kind"] == "positive"
            variant = get(entry, "variant", nothing)
            variant in KNOWN_VARIANTS ||
                (println(stderr, "$name: unknown positive variant $variant"); exit(1))
        end
        path = joinpath(root, String(entry["file"]))
        isfile(path) || fail(name, "missing file $(entry["file"])")
        digest = bytes2hex(sha256(read(path)))
        digest == String(entry["sha256"]) ||
            fail(name, "sha256 drift: $digest != $(entry["sha256"])")
    end
    covered = Set(String(entry["variant"]) for entry in fixtures if entry["kind"] == "positive")
    missing = [variant for variant in KNOWN_VARIANTS if !(variant in covered)]
    isempty(missing) || (println(stderr, "missing positive vector for $missing"); exit(1))
    return fixtures
end

function check_fixture(root::AbstractString, entry)
    name = String(entry["name"])
    blob = read(joinpath(root, String(entry["file"])))
    expect = String(entry["expect"])
    message = try
        decode_message(blob)
    catch err
        err isa WireError || rethrow()
        if expect == "decode_ok"
            fail(name, "expected decode_ok, got $(err.kind): $(err.reason)")
        end
        err.kind == expect || fail(name, "expected $expect, got $(err.kind): $(err.reason)")
        isempty(strip(err.reason)) && fail(name, "rejection reason is empty")
        return nothing
    end
    expect == "decode_ok" || fail(name, "expected $expect, but decode succeeded as $(message.variant)")
    String(entry["kind"]) in DECODE_OK_KINDS || fail(name, "unexpected kind $(entry["kind"])")
    if entry["kind"] == "positive"
        message.variant == String(entry["variant"]) ||
            fail(name, "variant $(message.variant) != $(entry["variant"])")
        encoded = encode_canonical(message)
        if encoded != blob
            fail(
                name,
                "canonical re-encode drifted\n  got $(bytes_string(encoded))\n  want $(bytes_string(blob))",
            )
        end
        return nothing
    end
    encoded = encode_canonical(message)
    again = decode_message(encoded)
    encode_canonical(again) == encoded || fail(name, "normalized re-encode is not stable")
    encoded == blob &&
        fail(name, "hand-maintained input was already canonical; check the fixture kind")
    return nothing
end

function decode_error(root::AbstractString, entry)::WireError
    try
        decode_message(read(joinpath(root, String(entry["file"]))))
    catch err
        err isa WireError && return err
        rethrow()
    end
    fail(String(entry["name"]), "expected a rejection")
end

function check_named_cases(root::AbstractString, fixtures)
    by_name = Dict(String(entry["name"]) => entry for entry in fixtures)
    classify_wire_version(0) == "too_old" || fail("classify", "version 0 must be too_old")
    classify_wire_version(1) == "supported" || fail("classify", "version 1 must be supported")
    classify_wire_version(2) == "too_new" || fail("classify", "version 2 must be too_new")

    too_new = decode_error(root, by_name["v2_too_new_malformed_payload"])
    if too_new.kind != "too_new" || !occursin("payload was not interpreted", too_new.reason)
        fail("v2_too_new_malformed_payload", "version must win, got $too_new")
    end
    occursin("2", too_new.reason) ||
        fail("v2_too_new_malformed_payload", "reason must name the found version")

    unknown = decode_error(root, by_name["v1_unknown_variant"])
    if unknown.kind != "payload_error" || !occursin("NotARealVariant", unknown.reason)
        fail("v1_unknown_variant", "unexpected rejection $unknown")
    end

    wrong = decode_error(root, by_name["v1_wrong_tagging"])
    if wrong.kind != "payload_error" || !occursin("internally tagged", wrong.reason)
        fail("v1_wrong_tagging", "unexpected rejection $wrong")
    end

    internal = Vector{UInt8}(
        "{\"wire_version\":1,\"payload\":{\"type\":\"Stimuli\",\"values\":[0.0],\"valid_mask\":null}}",
    )
    try
        decode_message(internal)
        fail("internal_stimuli", "internally tagged Stimuli was accepted")
    catch err
        err isa WireError || rethrow()
        if err.kind != "payload_error" || !occursin("internally tagged", err.reason)
            fail("internal_stimuli", "unexpected rejection $err")
        end
    end

    alias_bytes = read(joinpath(root, String(by_name["alias_neuron_id_traces"]["file"])))
    alias_text = bytes_string(alias_bytes)
    if !occursin("neuron_id", alias_text) || occursin("channel_id", alias_text)
        fail("alias_neuron_id_traces", "fixture must use the neuron_id alias only")
    end
    alias = decode_message(alias_bytes)
    if !(alias.body isa TraceBatch) || alias.body.traces[1].channel_id != 5
        fail("alias_neuron_id_traces", "neuron_id did not decode as channel_id 5")
    end
    alias_canonical = bytes_string(encode_canonical(alias))
    if occursin("neuron_id", alias_canonical) || !occursin("\"channel_id\":5", alias_canonical)
        fail("alias_neuron_id_traces", "canonical output must use channel_id")
    end

    norm_bytes = read(joinpath(root, String(by_name["normalization_integer_config"]["file"])))
    occursin("16777217", bytes_string(norm_bytes)) ||
        fail("normalization_integer_config", "fixture must keep the literal 2^24+1 input")
    norm = decode_message(norm_bytes)
    norm.body isa ConfigPayload || fail("normalization_integer_config", "expected ConfigUpdate")
    big = norm.body.config["big"]
    small = norm.body.config["n"]
    (big.kind == :float && small.kind == :float) ||
        fail("normalization_integer_config", "JSON integers must decode as floats")
    if big.value != 16_777_216.0 || Int(big.value) == 16_777_217
        fail("normalization_integer_config", "big decoded as $(big.value)")
    end
    small.value == 42.0 || fail("normalization_integer_config", "n decoded as $(small.value)")
    norm_text = bytes_string(encode_canonical(norm))
    if occursin("16777217", norm_text) || !occursin("\"big\":16777216.0", norm_text)
        fail("normalization_integer_config", "canonical floats drifted: $norm_text")
    end
    occursin("\"n\":42.0", norm_text) ||
        fail("normalization_integer_config", "42 must be a float: $norm_text")

    signed = decode_message(read(joinpath(root, String(by_name["signed_zero"]["file"]))))
    signed.body isa ConfigPayload || fail("signed_zero", "expected ConfigUpdate")
    neg = signed.body.config["neg"].value
    pos = signed.body.config["pos"].value
    if signbit(neg) != true || signbit(pos) != false
        fail("signed_zero", "signs were $neg $pos")
    end

    masked = decode_message(read(joinpath(root, String(by_name["stimuli_masked"]["file"]))))
    unmasked = decode_message(read(joinpath(root, String(by_name["stimuli_unmasked"]["file"]))))
    if !(masked.body isa StimulusBatch) || masked.body.valid_mask != [true, false, true]
        fail("stimuli_masked", "valid_mask did not round-trip")
    end
    masked.body.values[2] == 0.0 || fail("stimuli_masked", "masked placeholder value changed")
    if !(unmasked.body isa StimulusBatch) || unmasked.body.valid_mask !== nothing
        fail("stimuli_unmasked", "null valid_mask must decode as absent")
    end
    mask_err = decode_error(root, by_name["v1_invalid_stimulus_mask"])
    if mask_err.kind != "payload_error" || !occursin("valid_mask", mask_err.reason)
        fail("v1_invalid_stimulus_mask", "unexpected rejection $mask_err")
    end

    neuro_err = decode_error(root, by_name["v1_invalid_neuromod_range"])
    if neuro_err.kind != "payload_error" || !occursin("dopamine", neuro_err.reason)
        fail("v1_invalid_neuromod_range", "unexpected rejection $neuro_err")
    end

    legacy_ping = decode_message(read(joinpath(root, String(by_name["legacy_unversioned_ping"]["file"]))))
    legacy_ping.variant == "Ping" || fail("legacy_unversioned_ping", "expected Ping")
    ping_bytes = read(joinpath(root, String(by_name["ping"]["file"])))
    encode_canonical(legacy_ping) == ping_bytes ||
        fail("legacy_unversioned_ping", "normalized Ping must match canonical ping.json")

    legacy_spikes = decode_message(
        read(joinpath(root, String(by_name["legacy_unversioned_spikes"]["file"]))),
    )
    legacy_spikes.body isa SpikeBatch || fail("legacy_unversioned_spikes", "expected Spikes")
    batch = legacy_spikes.body
    if batch.session_id != "sess-1" ||
       batch.batch_id != 7 ||
       batch.metadata !== nothing ||
       length(batch.spikes) != 1 ||
       batch.spikes[1].channel != 3
        fail("legacy_unversioned_spikes", "decoded fields do not match the legacy document")
    end
    startswith(bytes_string(encode_canonical(legacy_spikes)), "{\"payload\":") ||
        fail("legacy_unversioned_spikes", "normalized form must be a v1 envelope")
    return nothing
end

function check_numeric_contract()
    uint_to_f32(16_777_217) == 16_777_216.0 ||
        (println(stderr, "16777217 must round to f32 16777216"); exit(1))
    f32_max = Float64(floatmax(Float32))
    expected = [
        (1e-6, "1e-6"),
        (1e-5, "0.00001"),
        (1e-4, "0.0001"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (16_777_216.0, "16777216.0"),
        (42.0, "42.0"),
        (-0.0, "-0.0"),
        (f32_max, "3.4028234663852886e+38"),
        (Float64(Float32(0.8)), "0.800000011920929"),
        (Float64(Float32(1e-6)), "9.999999974752427e-7"),
        (Float64(Float32(1e-5)), "9.999999747378752e-6"),
        (Float64(Float32(1e20)), "1.0000000200408773e+20"),
    ]
    for (value, text) in expected
        got = format_f64(value)
        got == text || (println(stderr, "float format $value: got $got, want $text"); exit(1))
    end
    return nothing
end

function check_missing_optionals()
    spikes = decode_message(Vector{UInt8}("{\"Spikes\":{\"batch_id\":1,\"timestamp\":2,\"spikes\":[]}}"))
    spikes.body isa SpikeBatch && spikes.body.metadata === nothing ||
        fail("missing_spike_metadata", "missing metadata must decode as null")
    occursin("\"metadata\":null", bytes_string(encode_canonical(spikes))) ||
        fail("missing_spike_metadata", "canonical output must include metadata null")
    partial = Vector{UInt8}(
        "{\"wire_version\":1,\"payload\":{\"Spikes\":{\"batch_id\":1,\"timestamp\":2,\"spikes\":[],\"metadata\":{\"source\":\"encoder\"}}}}",
    )
    decoded = decode_message(partial)
    decoded.body isa SpikeBatch && decoded.body.metadata !== nothing ||
        fail("missing_latency", "partial metadata must still decode")
    decoded.body.metadata.processing_latency_ns === nothing ||
        fail("missing_latency", "missing processing_latency_ns must be null")
    decoded.body.metadata.source == "encoder" ||
        fail("missing_latency", "source must still decode")
    stimuli = Vector{UInt8}(
        "{\"wire_version\":1,\"payload\":{\"Stimuli\":{\"session_id\":null,\"batch_id\":1,\"timestamp\":2,\"values\":[0.0],\"valid_mask\":null}}}",
    )
    stim = decode_message(stimuli)
    stim.body isa StimulusBatch && stim.body.metadata === nothing ||
        fail("missing_stimulus_metadata", "missing metadata must decode as null")
    try
        decode_message(Vector{UInt8}(
            "{\"wire_version\":1,\"payload\":{\"Stimuli\":{\"batch_id\":1,\"timestamp\":2,\"values\":[0.0],\"session_id\":null}}}",
        ))
        fail("missing_valid_mask", "expected a rejection")
    catch err
        err isa WireError || rethrow()
        (err.kind == "payload_error" && occursin("valid_mask", err.reason)) ||
            fail("missing_valid_mask", "unexpected rejection $err")
    end
    return nothing
end

function main()
    check_numeric_contract()
    check_missing_optionals()
    root = vector_dir()
    manifest = load_manifest(root)
    fixtures = check_manifest_hashes(root, manifest)
    for entry in fixtures
        check_fixture(root, entry)
    end
    check_named_cases(root, fixtures)
    println("ok $(length(fixtures)) wire-v1 fixtures")
    return nothing
end

main()
