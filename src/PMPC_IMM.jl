module PMPC_IMM

import Parameters
import LinearAlgebra
import DifferentialEquations
import Distributions
import JuMP

module PMPC
include(joinpath("PMPC", "PMPC_Jump.jl"))
export umpc, IMM, ssModel, PMPCSetup, belief, genGmat!
end

module Hexacopter
include(joinpath("Hexacopter", "HexacopterLinear.jl"))
export LinearModel
export simulate_nonlinear
export MixMat
end

module Cartpole
include(joinpath("Cartpole", "CartpoleLinear.jl"))
export LinearModel
export simulate_nonlinear, f!
end

end # module
