using Printf, LinearAlgebra
using CairoMakie
using CUDA
using Enzyme

macro d_xa(A) esc(:($A[ix+1, iz] - $A[ix, iz])) end
macro d_za(A) esc(:($A[ix, iz+1] - $A[ix, iz])) end
macro avx(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix+1, iz]))) end
macro avz(A)  esc(:(0.5 * ($A[ix, iz] + $A[ix, iz+1]))) end
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avz(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])
@views maxloc(A) = max.(A[2:end-1, 2:end-1], max.(max.(A[1:end-2, 2:end-1], A[3:end, 2:end-1]),
                                                  max.(A[2:end-1, 1:end-2], A[2:end-1, 3:end])))

function smooth_d!(A2, A)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iz = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if (ix>1 && ix<size(A, 1) && iz>1 && iz<size(A, 2))
        @inbounds A2[ix, iz] = A[ix, iz] + 0.2 * (A[ix+1, iz] - 2A[ix, iz] + A[ix-1, iz] + A[ix, iz+1] - 2A[ix, iz] + A[ix, iz-1])
    end
    return
end

function smooth!(A2, A, nthread, nblock; nsm=1)
    for _ ∈ 1:nsm
        CUDA.@sync @cuda threads=nthread blocks=nblock smooth_d!(A2, A)
        A, A2 = A2, A
    end
    return
end

# forward
function residual_fluxes!()
    #= ??? =#
    return
end

function residual_pressure!()
    #= ??? =#
    return
end

function update_fluxes!()
    #= ??? =#
    return
end

function update_pressure!()
    #= ??? =#
    return
end

@inline ∇(fun,args...) = (Enzyme.autodiff_deferred(Enzyme.Reverse, fun, args...); return)
const DupNN = DuplicatedNoNeed

@views function forward_solve!(logK, fields, scalars, iter_params; visu=nothing)
    (;Pf, qx, qz, Qf, RPf, Rqx, Rqz, K)               = fields
    (;nx, nz, dx, dz, nthread, nblock)                = scalars
    (;cfl, re, vdτ, lz, ϵtol, maxiter, ncheck, K_max) = iter_params
    isnothing(visu) || ((;qx_c, qz_c, qM, fig, plt, st) = visu)
    K .= exp.(logK)
    # approximate diagonal (Jacobi) preconditioner
    K_max .= K; K_max[2:end-1, 2:end-1] .= maxloc(K); K_max[:, [1, end]] .= K_max[:, [2, end-1]]
    # iterative loop
    iters_evo = Float64[]; errs_evo = Float64[]
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        #= ??? =#
        #= ??? =#
        #= ??? =#
        #= ??? =#
        if iter % ncheck == 0
            err = maximum(abs.(RPf))
            push!(iters_evo, iter/nx); push!(errs_evo, err)
            @printf("  #iter/nx=%.1f, max(err)=%1.3e\n", iter/nx, err)
            if !isnothing(visu)
                qx_c .= Array(avx(qx)); qz_c .= Array(avz(qz)); qM .= sqrt.(qx_c.^2 .+ qz_c.^2)
                qx_c ./= qM; qz_c ./= qM
                plt.fld.Pf[3] = Array(Pf)
                plt.fld.K[3]  = Array(log10.(K))
                plt.fld.qM[3] = qM
                plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
                plt.fld.ar[4] = qz_c[1:st:end, 1:st:end]
                plt.err[1] = Point2.(iters_evo, errs_evo)
                display(fig)
            end
        end
        iter += 1
    end
    return
end

@views function adjoint_solve!(logK, fwd_params, adj_params, loss_params)
    # unpack forward
    (;Pf, qx, qz, Qf, RPf, Rqx, Rqz, K) = fwd_params.fields
    (;nx, nz, dx, dz, nthread, nblock)  = fwd_params.scalars
    # unpack adjoint
    (;P̄f, q̄x, q̄z, R̄Pf, R̄qx, R̄qz, Ψ_qx, Ψ_qz, Ψ_Pf)      = adj_params.fields
    (;∂J_∂Pf)                                           = loss_params.fields
    (;cfl, re_a, vdτ, lz, ϵtol, maxiter, ncheck, K_max) = adj_params.iter_params
    # iterative loop
    iters_evo = Float64[]; errs_evo = Float64[]
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        R̄qx .= Ψ_qx
        R̄qz .= Ψ_qz
        P̄f  .= #= ??? =#
        q̄x  .= 0.0
        q̄z  .= 0.0
        #= ??? =#
        #= ??? =#
        #= ??? =#
        R̄Pf .= #= ??? =#
        #= ??? =#
        #= ??? =#
        if iter % ncheck == 0
            err = maximum(abs.(P̄f))
            push!(iters_evo, iter/nx); push!(errs_evo, err)
            @printf("  #iter/nx=%.1f, max(err)=%1.6e\n", iter/nx, err)
        end
        iter += 1
    end
    return
end

@views function loss(logK, fwd_params, loss_params; kwargs...)
    (;Pf_obs)       = loss_params.fields
    (;ixobs, izobs) = loss_params.scalars
    @info "Forward solve"
    #= ??? =#
    Pf = fwd_params.fields.Pf
    return 0.5*sum((Pf[ixobs, izobs] .- Pf_obs).^2)
end

function ∇loss!(logK̄, logK, fwd_params, adj_params, loss_params; reg=nothing, kwargs...)
    # unpack
    (;R̄qx, R̄qz, Ψ_qx, Ψ_qz)    = adj_params.fields
    (;Pf, qx, qz, Rqx, Rqz, K) = fwd_params.fields
    (;dx, dz, nthread, nblock) = fwd_params.scalars
    (;Pf_obs, ∂J_∂Pf)          = loss_params.fields
    (;ixobs, izobs)            = loss_params.scalars
    @info "Forward solve"
    #= ??? =#
    # set tangent
    ∂J_∂Pf[ixobs, izobs] .= #= ??? =#
    @info "Adjoint solve"
    #= ??? =#
    # evaluate gradient dJ_dK
    R̄qx .= .-Ψ_qx
    R̄qz .= .-Ψ_qz
    logK̄ .= 0.0
    #= ??? =#
    # Tikhonov regularisation (smoothing)
    if !isnothing(reg)
        (;nsm, Tmp) = reg
        Tmp .= logK̄; smooth!(logK̄, Tmp, nthread, nblock; nsm)
    end
    logK̄ .*= K # convert to dJ_dlogK by chain rule
    return
end

@views function main()
    CUDA.device!(0) # select your GPU
    # physics
    lx, lz  = 2.0, 1.0 # domain extend
    k0_μ    = 1.0      # background permeability / fluid viscosity
    kb_μ    = 1e-6     # barrier permeability / fluid viscosity
    Q_in    = 1.0      # injection flux
    b_w     = 0.02lx   # barrier width
    b_b     = 0.3lz    # barrier bottom location
    b_t     = 0.8lz    # barrier top location
    # observations
    xobs_rng = LinRange(-lx / 6, lx / 6, 8)
    zobs_rng = LinRange(0.25lz, 0.85lz , 8)
    # numerics
    nz       = 255
    nx       = ceil(Int, (nz + 1) * lx / lz) - 1
    nthread  = (16, 16)
    nblock   = cld.((nx, nz), nthread)
    cfl      = 1 / 2.1
    ϵtol     = 1e-6
    maxiter  = 30nx
    ncheck   = 2nx
    re       = 0.8π # fwd re
    st       = ceil(Int, nx / 30)
    # GD params
    ngd      = 50
    Δγ       = 0.2
    # preprocessing
    re_a     = 2re  # adjoint re
    dx, dz   = lx / nx, lz / nz
    xc, zc   = LinRange(-lx / 2 + dx / 2, lx / 2 - dx / 2, nx), LinRange(dz / 2, lz - dz / 2, nz)
    vdτ      = cfl * min(dx, dz)
    ixobs    = floor.(Int, (xobs_rng .- xc[1]) ./ dx) .+ 1
    izobs    = floor.(Int, (zobs_rng .- zc[1]) ./ dz) .+ 1
    # init
    Pf       = CUDA.zeros(Float64, nx, nz)
    RPf      = CUDA.zeros(Float64, nx, nz)
    qx       = CUDA.zeros(Float64, nx + 1, nz)
    Rqx      = CUDA.zeros(Float64, nx + 1, nz)
    qz       = CUDA.zeros(Float64, nx, nz + 1)
    Rqz      = CUDA.zeros(Float64, nx, nz + 1)
    Qf       = CUDA.zeros(Float64, nx, nz)
    K        = k0_μ .* CUDA.ones(Float64, nx, nz)
    logK     = CUDA.zeros(Float64, nx, nz)
    Tmp      = CUDA.zeros(Float64, nx, nz)
    # init adjoint storage
    Ψ_qx     = CUDA.zeros(Float64, nx + 1, nz)
    q̄x       = CUDA.zeros(Float64, nx + 1, nz)
    R̄qx      = CUDA.zeros(Float64, nx + 1, nz)
    Ψ_qz     = CUDA.zeros(Float64, nx, nz + 1)
    q̄z       = CUDA.zeros(Float64, nx, nz + 1)
    R̄qz      = CUDA.zeros(Float64, nx, nz + 1)
    Ψ_Pf     = CUDA.zeros(Float64, nx, nz)
    P̄f       = CUDA.zeros(Float64, nx, nz)
    R̄Pf      = CUDA.zeros(Float64, nx, nz)
    ∂J_∂Pf   = CUDA.zeros(Float64, nx, nz)
    dJ_dlogK = CUDA.zeros(Float64, nx, nz)
    # set low permeability barrier location
    K[ceil(Int, (lx/2-b_w)/dx):ceil(Int, (lx/2+b_w)/dx), ceil(Int, b_b/dz):ceil(Int, b_t/dz)] .= kb_μ
    logK .= log.(K)
    K_max = copy(K)
    # set wells location
    x_iw, x_ew, z_w = ceil.(Int, (lx / 5 / dx, 4lx / 5 / dx, 0.45lz / dz))
    Qf[x_iw:x_iw, z_w:z_w] .=  Q_in / dx / dz # injection
    Qf[x_ew:x_ew, z_w:z_w] .= -Q_in / dx / dz # extraction
    # init visu
    iters_evo = Float64[]; errs_evo = Float64[]
    qM, qx_c, qz_c = zeros(nx, nz), zeros(nx, nz), zeros(nx, nz)
    fig = Figure(resolution=(2500, 1200), fontsize=32)
    ax = ( Pf  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), title="Pf"),
           K   = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="log10(K)"),
           qM  = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), title="|q|"),
           err = Axis(fig[2, 2]; yscale=log10, title="Convergence", xlabel="# iter/nx", ylabel="error"), )
    plt = (fld = ( Pf   = heatmap!(ax.Pf, xc, zc, Array(Pf); colormap=:turbo, colorrange=(-1,1)),
                   K    = heatmap!(ax.K , xc, zc, Array(log10.(K)); colormap=:turbo, colorrange=(-6,0)),
                   xobs = scatter!(ax.K , vec(Point2.(xobs_rng, zobs_rng')); color=:white),
                   qM   = heatmap!(ax.qM, xc, zc, qM; colormap=:turbo, colorrange=(0,30)),
                   ar   = arrows!(ax.Pf, xc[1:st:end], zc[1:st:end], qx_c[1:st:end, 1:st:end], qz_c[1:st:end, 1:st:end]; lengthscale=0.05, arrowsize=15), ),
           err = scatterlines!(ax.err, Point2.(iters_evo, errs_evo), linewidth=4), )
    Colorbar(fig[1, 1][1, 2], plt.fld.Pf)
    Colorbar(fig[1, 2][1, 2], plt.fld.K)
    Colorbar(fig[2, 1][1, 2], plt.fld.qM)
    # action
    fwd_params = (
        fields      = (;Pf, qx, qz, Qf, RPf, Rqx, Rqz, K),
        scalars     = (;nx, nz, dx, dz, nthread, nblock),
        iter_params = (;cfl, re, vdτ, lz, ϵtol, maxiter, ncheck, K_max),
    )
    fwd_visu = (;qx_c, qz_c, qM, fig, plt, st)
    @info "Synthetic solve"
    #= ??? =#
    # store true data
    Pf_obs = copy(Pf[ixobs, izobs])
    adj_params = (
        fields  = (;P̄f, q̄x, q̄z, R̄Pf, R̄qx, R̄qz, Ψ_qx, Ψ_qz, Ψ_Pf),
        iter_params = (;cfl, re_a, vdτ, lz, ϵtol, maxiter, ncheck, K_max),
    )
    loss_params = (
        fields  = (;Pf_obs, ∂J_∂Pf),
        scalars = (;ixobs, izobs),
    )
    reg = (;nsm=20, Tmp)
    # loss functions
    J(_logK) = loss(_logK, fwd_params, loss_params)
    ∇J!(_logK̄, _logK) = ∇loss!(_logK̄, _logK, fwd_params, adj_params, loss_params; reg)
    @info "Inversion for K"
    # initial guess
    K    .= k0_μ
    logK .= log.(K)
    @info "Gradient descent - inversion for K"
    cost_evo = Float64[]
    for igd in 1:ngd
        printstyled("> GD iter $igd \n"; bold=true, color=:green)
        # evaluate gradient of the cost function
        #= ??? =#
        # update logK
        γ = Δγ / maximum(abs.(dJ_dlogK))
        @. logK -= #= ??? =#
        @printf "  min(K) = %1.2e \n" minimum(K)
        # loss
        push!(cost_evo, J(logK))
        @printf "  --> Loss J = %1.2e (γ = %1.2e)\n" last(cost_evo)/first(cost_evo) γ
        # visu
        qx_c .= Array(avx(qx)); qz_c .= Array(avz(qz)); qM .= sqrt.(qx_c.^2 .+ qz_c.^2)
        qx_c ./= qM; qz_c ./= qM
        plt.fld.Pf[3] = Array(Pf)
        plt.fld.K[3]  = Array(log10.(K))
        plt.fld.qM[3] = qM
        plt.fld.ar[3] = qx_c[1:st:end, 1:st:end]
        plt.fld.ar[4] = qz_c[1:st:end, 1:st:end]
        plt.err[1] = Point2.(1:igd, cost_evo ./ 0.999cost_evo[1])
        display(fig)
    end
    return
end

main()