module HexarotorMPC

import Parameters
import LinearAlgebra
import DifferentialEquations

include(join("Hexacopter", "HexacopterLinear.jl"))
export LinearModel
export simulate_nonlinear

include(joinpath("Cartpole", "CartpoleLinear.jl"))
export LinearModelCartPole
export simulate_nonlinear_cartpole

end # module
