using Markdown
md"""
# GPU perftest 2D

Load modules
"""
using CUDA
using BenchmarkTools

md"""
FD derivatives macros
"""
macro d2_xi(A) esc(:(($A[ix+2, iy+1] - $A[ix+1, iy+1]) - ($A[ix+1, iy+1] - $A[ix, iy+1]))) end
macro d2_yi(A) esc(:(($A[ix+1, iy+2] - $A[ix+1, iy+1]) - ($A[ix+1, iy+1] - $A[ix+1, iy]))) end
macro all(A)  esc(:($A[ix, iy])) end
macro inn(A)  esc(:($A[ix+1, iy+1])) end

md"""
Memory copy "saxpy" function
"""
function memcopy!(C2, C, D, dt)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (ix <= size(C, 1) && iy <= size(C, 2))
        @inbounds @all(C2) = @all(C) + dt * @all(D)
    end
    return
end

md"""
Laplacian or diffusion step function
"""
function diffusion_step!(C2, C, D, dt, _dx, _dy)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (ix <= size(C, 1) - 2 && iy <= size(C, 2) - 2)
        @inbounds @inn(C2) = @inn(C) + dt * @inn(D) * (@d2_xi(C) * _dx * _dx + @d2_yi(C) * _dy * _dy)
    end
    return
end

md"""
Performance test "lab"
"""
function perftest()
    nx = ny = 512 * 32 # 512 * 64
    C  = CUDA.rand(Float64, nx, ny)
    D  = CUDA.rand(Float64, nx, ny)
    _dx = _dy = dt = rand()
    C2 = copy(C)
    nthreads = (16, 16)
    nblocks  = cld.((nx, ny), nthreads)
    ## memory copy
    t_it_mcpy = @belapsed begin
        CUDA.@sync @cuda threads=$nthreads blocks=$nblocks memcopy!($C2, $C, $D, $dt)
    end
    T_peak = (2 * 1 + 1) / 1e9 * nx * ny * sizeof(Float64) / t_it_mcpy
    ## Laplacian
    t_it = @belapsed begin
        CUDA.@sync @cuda threads=$nthreads blocks=$nblocks diffusion_step!($C2, $C, $D, $dt, $_dx, $_dy)
    end
    T_eff = (2 * 1 + 1) / 1e9 * nx * ny * sizeof(Float64) / t_it
    println("T_eff = $(round(T_eff, sigdigits=6)) GiB/s (T_peak = $(round(T_peak, sigdigits=6)) GiB/s) using CUDA.jl on an Nvidia GPU")
    println("So that's cool. We are running at $(round(T_eff/T_peak*100, sigdigits=4)) % of memory copy (T_peak)! 🚀")
    return
end

md"""
Run the perf test
"""
perftest()
