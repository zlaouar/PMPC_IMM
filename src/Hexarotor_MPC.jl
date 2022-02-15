module Hexarotor_MPC

import Parameters
import LinearAlgebra
import DifferentialEquations

include("HexacopterLinear.jl")
export LinearModel
export simulate_nonlinear

end # module
