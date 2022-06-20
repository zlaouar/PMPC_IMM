module PMPC_IMM

using Parameters
using LinearAlgebra
using DifferentialEquations
using Distributions
using JuMP

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

include(joinpath("PMPC", "PMPC_Jump.jl"))
#export umpc, IMM, ssModel, ssModelm, PMPCSetup, belief, genGmat!
#export unom_vec, ss_params, Wd, Vd
include(joinpath("Estimator", "Estimator.jl"))
#export beliefUpdater, nl_dyn
#export nl_dyn_proc_noise, nl_dyn_meas_noise, nl_dyn_all_noise

end # module
