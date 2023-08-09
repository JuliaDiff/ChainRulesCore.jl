using Preferences

const int2float_type = @load_preference("int2float", "Float64")

function set_int2float!(int2float_type::String)
    if !(int2float_type in ("Float64", "Float32", "Float16"))
        throw(ArgumentError("Invalid int2float type: \"$(int2float_type)\""))
    end

    # Set it in our runtime values, as well as saving it to disk
    @set_preferences!("int2float" => int2float_type)
    @info(
        "New int2float type set; restart your Julia session for this change to take effect!"
    )
end
