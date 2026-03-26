#!/usr/bin/env julia
"""
Harvest I/O fixtures from Julia upstream repos.

Instruments functions via method-table overwriting, runs the repo's
test/runtests.jl, and writes captured call records to JSON.

Usage:
    julia scripts/harvest_jl.jl                        # all Julia entries
    julia scripts/harvest_jl.jl --repo Tempo.jl        # single repo
"""

using JSON3
using YAML

const ROOT = dirname(dirname(@__FILE__))
const MANIFEST_PATH = joinpath(ROOT, "scripts", "atom_manifest.yml")
const FIXTURES_DIR = joinpath(ROOT, "tests", "fixtures")
const THIRD_PARTY = joinpath(ROOT, "third_party")
const MAX_RECORDS = 10

# ── Serialization ────────────────────────────────────────────────────────────

function serialize_value(x::AbstractArray{T}) where T <: Number
    Dict(
        "__ndarray__" => true,
        "data" => base64encode(reinterpret(UInt8, vec(x))),
        "dtype" => string(T),
        "shape" => collect(size(x)),
    )
end

serialize_value(x::Number) = x
serialize_value(x::AbstractString) = x
serialize_value(x::Bool) = x
serialize_value(::Nothing) = nothing
serialize_value(x::Tuple) = Dict("__tuple__" => true, "items" => [serialize_value(v) for v in x])
serialize_value(x::Dict) = Dict(string(k) => serialize_value(v) for (k, v) in x)

function serialize_value(x)
    try
        Dict("__julia_repr__" => true, "type" => string(typeof(x)), "repr" => repr(x))
    catch
        nothing
    end
end

using Base64

# ── Manifest loading ─────────────────────────────────────────────────────────

function load_julia_entries(; repo_filter=nothing)
    manifest = YAML.load_file(MANIFEST_PATH)
    entries = filter(manifest) do entry
        up = entry["upstream"]
        up["language"] == "julia" &&
            (repo_filter === nothing || up["repo"] == repo_filter)
    end
    return entries
end

# ── Recording wrapper ────────────────────────────────────────────────────────

mutable struct Recorder
    atom_key::String
    func_name::String
    records::Vector{Dict}
end

function wrap_function!(mod::Module, func_sym::Symbol, atom_key::String)
    recorder = Recorder(atom_key, string(func_sym), Dict[])

    original = getfield(mod, func_sym)

    wrapped = function(args...; kwargs...)
        result = original(args...; kwargs...)
        if length(recorder.records) < MAX_RECORDS
            try
                record = Dict(
                    "function" => recorder.func_name,
                    "atom" => recorder.atom_key,
                    "inputs" => Dict("args" => [serialize_value(a) for a in args]),
                    "output" => serialize_value(result),
                )
                push!(recorder.records, record)
            catch e
                @debug "Serialization failed for $(recorder.func_name): $e"
            end
        end
        return result
    end

    @eval mod const $(func_sym) = $wrapped

    return recorder
end

# ── Fixture I/O ──────────────────────────────────────────────────────────────

function fixture_path(atom_key::String)
    parts = split(atom_key, ":")
    module_part = parts[1]
    func_name = parts[2]
    return joinpath(FIXTURES_DIR, module_part, "$(func_name).json")
end

function save_fixture(records::Vector{Dict}, path::String)
    mkpath(dirname(path))
    open(path, "w") do f
        JSON3.write(f, records)
    end
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    repo_filter = nothing
    for (i, arg) in enumerate(ARGS)
        if arg == "--repo" && i < length(ARGS)
            repo_filter = ARGS[i + 1]
        end
    end

    entries = load_julia_entries(; repo_filter)
    if isempty(entries)
        @error "No matching Julia manifest entries found"
        return
    end

    @info "Harvesting $(length(entries)) Julia atom(s)"

    # Group by repo
    by_repo = Dict{String, Vector}()
    for entry in entries
        repo = entry["upstream"]["repo"]
        push!(get!(by_repo, repo, []), entry)
    end

    saved = 0
    for (repo, repo_entries) in by_repo
        repo_path = joinpath(THIRD_PARTY, repo)
        if !isdir(repo_path)
            @warn "Repo $repo not found at $repo_path"
            continue
        end

        # Add repo to LOAD_PATH
        push!(LOAD_PATH, joinpath(repo_path, "src"))

        recorders = Recorder[]
        for entry in repo_entries
            up = entry["upstream"]
            atom_key = entry["atom"]
            try
                mod = Base.require(Main, Symbol(up["module"]))
                recorder = wrap_function!(mod, Symbol(up["function"]), atom_key)
                push!(recorders, recorder)
                @info "Instrumented $(up["module"]).$(up["function"]) → $atom_key"
            catch e
                @warn "Cannot instrument $atom_key: $e"
            end
        end

        # Try running tests
        test_file = joinpath(repo_path, "test", "runtests.jl")
        if isfile(test_file)
            @info "Running tests for $repo"
            try
                include(test_file)
            catch e
                @warn "Test run failed for $repo: $e"
            end
        end

        # Save captured records
        for recorder in recorders
            if !isempty(recorder.records)
                path = fixture_path(recorder.atom_key)
                save_fixture(recorder.records, path)
                @info "Saved $(length(recorder.records)) records → $path"
                saved += 1
            end
        end

        # Clean LOAD_PATH
        filter!(p -> p != joinpath(repo_path, "src"), LOAD_PATH)
    end

    @info "Done: $saved fixture files written"
end

main()
