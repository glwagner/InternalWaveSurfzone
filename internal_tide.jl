using Oceananigans
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.Units
using Printf
using GLMakie

Nx, Nz = 250, 125
H = 100
L = 30kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  x = (-2L, L),
                                  z = (-H, 0),
                                  halo = (4, 4),
                                  topology = (Periodic, Flat, Bounded))

sloping_bottom(x) = - H * (1 - max(0, x / L))
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(sloping_bottom))

# lines(grid.immersed_boundary.bottom_height)
# current_figure()

coriolis = FPlane(latitude=35)
T₂ = 12.421hours
ω₂ = 2π / T₂ # radians/sec

m = π / H
N² = 1e-5
f = coriolis.f
k = m * sqrt((ω₂^2 - f^2) / N²)
@show λ = 2π/k
# k² = (ω² - f²) * m² / N²
δ = L/4
x₀ = -3L/2
parameters = (; k, ω=ω₂, m, δ, x₀, U=0.1)
@inline A(x, p) = exp(-(x - x₀)^2 / 2δ^2)
@inline igw_forcing(x, z, t, p) = p.ω * p.U * A(x, p) * cos(p.k * x - p.ω * t) * sin(p.m * z)
u_forcing = Forcing(igw_forcing; parameters)

model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                                    closure =  CATKEVerticalDiffusivity(),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :e),
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    forcing = (; u = u_forcing))

@info "Set up a model:"
@show model

bᵢ(x, z) = N² * z
set!(model, b=bᵢ)

Δt = 5minutes
stop_time = 4days
simulation = Simulation(model; Δt, stop_time)

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

b = model.tracers.b
u, v, w = model.velocities
U = Field(Average(u))
u′ = u - U
N² = ∂z(b)

filename = "internal_tide"
save_fields_interval = 30minutes
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, u′, w, b, N²);
                                                      filename,
                                                      schedule = TimeInterval(save_fields_interval),
                                                      overwrite_existing = true)

run!(simulation)

#=
saved_output_filename = filename * ".jld2"

u′_t = FieldTimeSeries(saved_output_filename, "u′")
 w_t = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")

umax = maximum(abs, u′_t[end])
wmax = maximum(abs, w_t[end])

times = u′_t.times

using CairoMakie

n = Observable(1)

title = @lift @sprintf("t = %1.2f days = %1.2f T₂",
                       round(times[$n] / day, digits=2) , round(times[$n] / T₂, digits=2))

u′ₙ = @lift u′_t[$n]
 wₙ = @lift  w_t[$n]
N²ₙ = @lift N²_t[$n]

axis_kwargs = (xlabel = "x [m]",
               ylabel = "z [m]",
               limits = ((-grid.Lx/2, grid.Lx/2), (-grid.Lz, 0)),
               titlesize = 20)

fig = Figure(size = (700, 900))

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u'-velocity", axis_kwargs...)
hm_u = heatmap!(ax_u, u′ₙ; nan_color=:gray, colorrange=(-umax, umax), colormap=:balance)
Colorbar(fig[2, 2], hm_u, label = "m s⁻¹")

ax_w = Axis(fig[3, 1]; title = "w-velocity", axis_kwargs...)
hm_w = heatmap!(ax_w, wₙ; nan_color=:gray, colorrange=(-wmax, wmax), colormap=:balance)
Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

ax_N² = Axis(fig[4, 1]; title = "stratification N²", axis_kwargs...)
hm_N² = heatmap!(ax_N², N²ₙ; nan_color=:gray, colorrange=(0.9Nᵢ², 1.1Nᵢ²), colormap=:magma)
Colorbar(fig[4, 2], hm_N², label = "s⁻²")

fig

# Finally, we can record a movie.

@info "Making an animation from saved data..."

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=16) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    n[] = i
end
nothing #hide

# ![](internal_tide.mp4)
=#
