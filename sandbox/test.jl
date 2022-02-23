using Distributions
using LinearAlgebra

LL = [-5668.929869349505, -59107.650511822125, -27280.06120792378, -39273.62411941404, -22391.02628823826, -24455.383923614343, -73338.51191685321]
maxL = maximum(LL)
@show logsumexp = LL .+ sum(exp.(LL .- maxL)) .- maxL
LL = exp.(logsumexp) / sum(exp.(logsumexp))

function testDist()
    dist = MvNormal(Diagonal([1,1,1]))
    print("hello")
end

testDist()