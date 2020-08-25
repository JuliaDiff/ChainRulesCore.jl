@testset "rule_definition_tools.jl" begin
    
    @testset "@nondifferentiable" begin

    end
end


Base.remove_linenums!(@macroexpand @non_differentiable println(io::IO))

@non_differentiable println(io::IO)