import julia
from julia import Main
import pickle

julia.install()

Main.eval("""
using Arpack
using LinearAlgebra
using StaticArrays
""")

Main.eval("""
# polar angle of point in range [0, 2 * pi)
function polar_angle(x, y)
    ang = 0
    if x == 0 && y == 0
        ang = 0
    elseif x < 0
        ang = atan(y/x) + pi
    else
        ang = atan(y/x)
    end
    if ang < 0
        ang += 2 * pi
    end
    return ang
end
function rot(pts, theta)
    new_points = Array{Float64}(undef, size(pts)[1], 2)
    for j in 1:size(pts)[1]
        x, y = pts[j, :]
        mag = norm([x, y])
        ang = polar_angle(x, y)
        new_points[j, 1] = mag * cos(ang + theta)
        new_points[j, 2] = mag * sin(ang + theta)
    end
    return new_points
end
function chunk_space(x0, x1, n)
    return collect(range(x0, x1, n + 2)[2:n + 1])
end
function stump_triangle(rad, spacing)
    counter = 0
    limit = Int(trunc(rad/spacing))
    x_coords = []
    y_coords = []
    counter = 0
    for j in 1:limit - 1
        if j == 1
            push!(x_coords, 0)
            push!(y_coords, -spacing * 2 * sin(2 * pi / 3))
            counter += 1
        else
            x_pts = chunk_space(spacing * (j + 1) * cos(2 * pi / 3), -spacing * (j + 1) * cos(2 * pi / 3), j)
            for i in 1:j
                push!(x_coords, x_pts[i])
                push!(y_coords, -spacing * (j + 1) * sin(2 * pi / 3))
                counter += 1
            end
        end
    end
    points = Array{Float64}(undef, counter, 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    return points
end
function hex_frame(rad, spacing)
    x_coords = []
    y_coords = []
    for j in -rad:spacing:rad
        if j != 0
            push!(x_coords, j * cos(0))
            push!(y_coords, j * sin(0))

            push!(x_coords, j * cos(2 * pi/3))
            push!(y_coords, j * sin(2 * pi / 3))

            push!(x_coords, j * cos(4 * pi/3))
            push!(y_coords, j * sin(4 * pi / 3))
        end
    end
    push!(x_coords, 0)
    push!(y_coords, 0)
    points = Array{Float64}(undef, size(x_coords)[1], 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    return points
end
function make_hex(center, rad, spacing)
    frame = hex_frame(rad, spacing)
    x_c = frame[:, 1]
    y_c = frame[:, 2]
    for j in 0:5
        tri = stump_triangle(rad, spacing)
        rot_pts = rot(tri, j * pi / 3)
        append!(x_c, rot_pts[:, 1])
        append!(y_c, rot_pts[:, 2])
    end
    points = Array{Float64}(undef, size(x_c)[1], 2)
    points[:, 1] = x_c .+ center[1] * ones(size(x_c)[1])
    points[:, 2] = y_c .+ center[2] * ones(size(y_c)[1])
    return points
end
# make the outline of a hexagon
function hex_outline(center, radius, spacing)
    cx, cy = center
    # Compute the 6 corners of the hexagon
    corners = [(cx + radius * cos(theta), cy + radius * sin(theta)) for theta in 0:pi/3:5pi/3]

    # Generate points along the 6 edges (including corners)
    frame_points = Set(corners)  # Use a Set to avoid duplicate corner points

    for i in 1:6
        p1 = corners[i]
        p2 = corners[mod1(i+1, 6)]  # Next corner (modulo 6)
        edge_vector = (p2[1] - p1[1], p2[2] - p1[2])
        edge_length = sqrt(edge_vector[1]^2 + edge_vector[2]^2)
        n_segments = max(1, round(Int, edge_length / spacing))

        for j in 1:(n_segments-1)
            t = j / n_segments
            px = (1 - t) * p1[1] + t * p2[1]
            py = (1 - t) * p1[2] + t * p2[2]
            push!(frame_points, (px, py))
        end
    end
    temp = collect(frame_points)
    
    grid = Array{Float64}(undef, size(temp, 1), 2)
    for j in 1:size(temp, 1)
        grid[j, 1] = temp[j][1]
        grid[j, 2] = temp[j][2]
    end
    return grid
end

# plaquette area for sidelength s, N-gon
function area(s, N)
    return (1/4) * N * s^2 * cot(pi/N)
end

function make_circle(kappa, spacing)
    x_coords = []
    y_coords = []
    num_rings = floor(Int, kappa / spacing)
    for i in 0:num_rings
        r = i * spacing
        num_points = max(1, floor(Int, 2 * pi * i))
        for j in 0:num_points-1
            theta = j * (2*pi / num_points)
            push!(x_coords, r * cos(theta))
            push!(y_coords, r * sin(theta))
        end
    end
    points = Array{Float64}(undef, size(x_coords, 1), 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    return points
end

function make_sector(r_min, r_max, spacing, theta_min, theta_max)
    x_coords = []
    y_coords = []
    num_rings = floor(Int, (r_max - r_min) / spacing)
    println(num_rings)
    
    for i in 0:num_rings
        r = i * spacing + r_min
        num_points = max(1, floor(Int, (theta_max - theta_min) * r / spacing))
        for j in 0:num_points-1
            theta = theta_min + j * ((theta_max - theta_min) / num_points)
            push!(x_coords, r * cos(theta))
            push!(y_coords, r * sin(theta))
        end
    end
    
    points = Array{Float64}(undef, length(x_coords), 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    
    return points
end


""")

Main.eval("""
# 2x2 Pauli X
function sigma_x()
    return [0 1; 1 0]
end
# 2x2 Pauli Y
function sigma_y()
    return [0 -im; im 0]
end
# 2x2 Pauli Z
function sigma_z()
    return [1 0; 0 -1]
end

function H_k(q, theta, vF)
    n1 = [cos(0), sin(0)]
    n3 = [cos(2 * pi / 3), sin(2 * pi / 3)]
    n5 = [cos(4 * pi / 3), sin(4 * pi / 3)]
    k = vF * q * [cos(theta), sin(theta)]
    return [dot(k, n1) 0 0; 0 dot(k, n3) 0; 0 0 dot(k, n5)]
end

function H_k_v2(k_m, vF)
    n1 = [cos(0), sin(0)]
    n3 = [cos(2 * pi / 3), sin(2 * pi / 3)]
    n5 = [cos(4 * pi / 3), sin(4 * pi / 3)]
    k = vF * k_m
    return [dot(k, n1) 0 0; 0 dot(k, n3) 0; 0 0 dot(k, n5)]
end
function H_mft(q, theta, delt, alph)
    # useful phase
    omega = exp(2*pi/3 * im)
    mat0 = [0 delt conj(delt); 
            conj(delt) 0 delt; 
            delt conj(delt) 0]
    qq = q * exp(theta * im)
    mat1 = [0 alph * (omega * qq + conj(omega) * conj(qq)) conj(alph) * (conj(omega) * qq + omega * conj(qq));
             conj(alph) * (omega * qq + conj(omega) * conj(qq)) 0 alph * (qq + conj(qq));
             alph * (conj(omega) * qq + omega * conj(qq)) conj(alph) * (qq + conj(qq)) 0]
    return mat0 + 1/2 * mat1
end

function H_mft_v2(k, delt, alph)
    # useful phase
    omega = exp(2*pi/3 * im)
    mat0 = [0 delt conj(delt); 
            conj(delt) 0 delt; 
            delt conj(delt) 0]
    qq = k[1] + im * k[2]
    mat1 = [0 alph * (omega * qq + conj(omega) * conj(qq)) conj(alph) * (conj(omega) * qq + omega * conj(qq));
             conj(alph) * (omega * qq + conj(omega) * conj(qq)) 0 alph * (qq + conj(qq));
             alph * (conj(omega) * qq + omega * conj(qq)) conj(alph) * (qq + conj(qq)) 0]
    return mat0 + 1/2 * mat1
end

function gauge_fix(state)
    entry = state[1]
    phi = angle(entry / abs(entry))
    state = state .* exp(-im * phi)
    return state
end

function multi_gauge_fix(states)
    # fix according to the phase of the 1st entry of the first state
    entry = states[1, 1]
    phi = angle(entry / abs(entry))
    len = size(states, 1)
    for j in 1:len
        states[j, :] = states[j, :] .* exp(-im * phi)
    end
    return states
end

function spinor_inner(C1, C2, q1_spinor, q2_spinor)
    val = 0
    norm1 = 0
    norm2 = 0
    spinor_inner!(val, norm1, norm2, C1, C2, q1_spinor, q2_spinor)
end

function spinor_inner!(val, n1, n2, C1, C2, q1_spinor, q2_spinor)
    for m in 1:length(C1)
        val += conj(C1[m]) * C2[m] * dot(q1_spinor[m, :], q2_spinor[m, :])
        n1 += conj(C1[m]) * C1[m] * dot(q1_spinor[m, :], q1_spinor[m, :])
        n2 += conj(C2[m]) * C2[m] * dot(q2_spinor[m, :], q2_spinor[m, :])
    end
    return val / sqrt(n1 * n2)
end

function ff_spinor_inner(C1, C2, form_factors)
    val = 0
    for m in 1:3
        val += conj(C1[m]) * C2[m] * form_factors[m]
    end
    return val
end

function bc_no_spinors(points, spacing, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        states = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            gs = eigvecs(Hermitian(ham))[:, 1]
            states[j, :] = gauge_fix(normalize(gs))
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = dot(states[j, :], states[j + 1, :])
                P *= temp
            else
                temp = dot(states[j, :], states[1, :])
                P *= temp
            end
            if temp != 0
                P /= abs(temp)
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / (area(spacing * sqrt(2), num_vertices))
        else
            berry_list[i] = -angle(P) / area(spacing * sqrt(2), num_vertices)
        end
    end
    return berry_list
end

function analytic_eigenvalues(alpha, delta, x, y)
    B = 6 * alpha * conj(alpha) * (x^2 + y^2) + 3 * delta * conj(delta)
    C = 4 * real(alpha^3) * x^3 - 6 * real(alpha^2 * delta) * x^2 - 6 * real(alpha^2 * delta) * y^2 - 
    12 * real(alpha^3) * x * y^2 + 2 * real(delta^3)
    epsilons = Array{Float64}(undef, 3)
    epsilons[3] = real(2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 0*2 * pi/3))
    epsilons[2] = real(2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 1*2 * pi/3))
    epsilons[1] = real(2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 2*2 * pi/3))
    return epsilons
end

function vF_analytic_eigenvalues(alpha, delta, x, y, vF)
    v = vF
    B = 3 * (x^2 + y^2) * (v^2 + 2 * abs2(alpha)) + 3 * abs2(delta)
    C = (2 * x * (v^3 - 3 * v * abs2(alpha) + 2 * real(alpha^3)) * (x^2 - 3 * y^2) - 
    6 * (x^2 + y^2) * real(delta * alpha^2 + 2 * v * alpha * conj(delta)) + 2 * real(delta^3))
    epsilons = Array{Float64}(undef, 3)
    epsilons[3] = 2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 0*2 * pi/3)
    epsilons[2] = 2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 1*2 * pi/3)
    epsilons[1] = 2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 2*2 * pi/3)
    return real.(epsilons)
end

# function vF_analytic_eigenvalues(alpha, delta, x, y, vF)
#     v = vF
#     B = -(3 * (x^2 + y^2) * (v^2 + 2 * abs2(alpha)) + 3 * abs2(delta))
#     C = (-(2 * x * (v^3 - 3 * v * abs2(alpha) + 2 * real(alpha^3)) * (x^2 - 3 * y^2) - 
#     6 * (x^2 + y^2) * real(delta * alpha^2 + 2 * v * alpha * conj(delta)) + 2 * real(delta^3)))
#     del = (C/2)^2 + (B/3)^3
#     u = (-C/2 + sqrt(Complex(del)))^(1/3)
#     w = (-C/2 - sqrt(Complex(del)))^(1/3)
#     omega = exp(im * 2 * pi/3)
#     epsilons = Array{Float64}(undef, 3)
#     epsilons[3] = real(u + w)
#     epsilons[1] = real(omega * u + conj(omega) * w)
#     epsilons[2] = real(conj(omega) * u + omega * w)
#     return epsilons
# end

function analytic_eigenvectors(epsilon, alpha, delta, x, y)
    # convenience
    q = x + im*y
    omega = exp(im * 2 * pi/ 3)
    # variables
    f1 = delta + alpha * (q + conj(q))
    f3 = delta + alpha * (omega * q + conj(omega * q))
    f5 = delta + alpha * (conj(omega) * q + omega * conj(q))
    # normalization
    nmz = (epsilon^6 + epsilon^4 * (abs2(f3) - 2 * abs2(f1) - 2 * abs2(f5)) - 2 * epsilon^3 * real(f1 * f3 * f5) + 
    epsilon^2 * (abs2(f1)^2 + abs2(f5)^2 + 2 * abs2(f1) * abs2(f5) - 2 * abs2(f1) * abs2(f3) + abs2(f3) * abs2(f5)) + 
    2 * epsilon * real(f1 * f3 * f5) * (abs2(f1) + abs2(f3) + abs2(f5)) + abs2(f1) * abs2(f3) * (abs2(f1) + abs2(f3) + abs2(f5)))
    # eigenvector entries
    A1 = abs(f3) * (epsilon^2 - abs2(f1))
    A3 = conj(f3)/abs(f3) * (epsilon * (epsilon^2 - abs2(f1)) - conj(f5) * (epsilon * f5 + conj(f1) * conj(f3)))
    A5 = abs(f3) * (epsilon * f5 + conj(f1) * conj(f3))
    return 1/sqrt(nmz) * [A1, A3, A5]
end

function vF_analytic_eigenvectors(epsilon, alpha, delta, x, y, vF)
    v = vF/2
    q = x + im * y
    omega = exp(im * 2 * pi/ 3)
    # variables
    f1 = delta + alpha * (q + conj(q))
    v1 = v * (q + conj(q))
    f3 = delta + alpha * (omega * q + conj(omega * q))
    v3 = v * (omega * q + conj(omega * q))
    f5 = delta + alpha * (conj(omega) * q + omega * conj(q))
    v5 = v * (conj(omega) * q + omega * conj(q))
    # normalization
    nmz = (((epsilon - v5) * (epsilon - v3) - abs2(f1))^2 + abs2(f1) * (abs2(f3) + abs2(f5)) + 
    2 * real(f1 * f3 * f5) * (2 * epsilon - v3 - v5) + abs2(f3) * (epsilon - v3)^2 + abs2(f5) * (epsilon - v5)^2)
    # eigenvector entries
    A1 = (epsilon - v5) * (epsilon - v3) - abs2(f1)
    A3 = conj(f3) * (epsilon - v3) + f1 * f5
    A5 = f5 * (epsilon - v5) + conj(f1) * conj(f3)
    return 1/sqrt(nmz) * [A1, A3, A5]
end

# At the origin
function analytic_d_A1(delta, alpha, index)
    omega = exp(im * 2 * pi/ 3)
    k = index - 2
    nmz = (64 * real(omega^k * delta)^6 - 48 * abs2(delta) * real(omega^k * delta)^4 - 16 * real(delta^3) * real(omega^k * delta)^3 + 
    12 * abs2(delta)^2 * real(omega^k * delta)^2 + 12 * abs2(delta) * real(delta^3) * real(omega^k * delta) + 3 * abs2(delta)^3)

    dx_A1 = 1/sqrt(nmz) * (-abs(delta)/(2 * nmz) * (-96 * real(omega^k * delta)^4 * real(alpha * conj(delta)) + 
    6 * abs2(delta)^2 * real(alpha * conj(delta))) * (4 * real(omega^k * delta)^2 - abs2(delta)) - 
    4 * abs(delta) * real(alpha * conj(delta)) - 
    real(alpha * conj(delta)) / abs(delta) * (4 * real(omega^k * delta)^2 - abs2(delta)))

    dy_A1 = 1/sqrt(nmz) * (-abs(delta)/(2 * nmz) * (-96*sqrt(3) * real(omega^k * delta)^4 * real(alpha * conj(delta)) + 
    48*sqrt(3) * abs2(delta) * real(omega^k * delta)^2 * real(alpha * conj(delta)) - 
    6*sqrt(3) * abs2(delta)^2 * real(alpha * conj(delta))) * (4 * real(omega^k * delta)^2 - abs2(delta)) - 
    sqrt(3)/abs(delta) * real(alpha * conj(delta)) * (4 * real(omega^k * delta)^2 - abs2(delta)))
    
    return dx_A1, dy_A1
end

# At the origin
function analytic_d_A3(delta, alpha, index)
    omega = exp(im * 2 * pi/ 3)
    k = index - 2

    nmz = (64 * real(omega^k * delta)^6 - 48 * abs2(delta) * real(omega^k * delta)^4 - 16 * real(delta^3) * real(omega^k * delta)^3 + 
    12 * abs2(delta)^2 * real(omega^k * delta)^2 + 12 * abs2(delta) * real(delta^3) * real(omega^k * delta) + 3 * abs2(delta)^3)
    
    dx_A3 = 1/sqrt(nmz) * (-conj(delta)/(2 * nmz * abs(delta)) * (-96 * real(omega^k * delta)^4 * real(alpha * conj(delta)) + 
    6 * abs2(delta)^2 * real(alpha * conj(delta))) * (8 * real(omega^k * delta)^3 - 4 * abs2(delta) * real(omega^k * delta) - conj(delta)^3) +
    (conj(delta)/abs(delta)^3 * real(alpha * conj(delta)) - conj(alpha)/abs(delta)) * 
    (8 * real(omega^k * delta)^3 - 4 * abs2(delta) * real(omega^k * delta) - conj(delta)^3) - 
    4 * conj(delta) / abs(delta) * real(omega^k * delta) * real(alpha * conj(delta)))

    dy_A3 = 1/sqrt(nmz) * (-conj(delta)/(2 * nmz * abs(delta)) * (-96 * sqrt(3) * real(omega^k * delta)^4 * real(conj(delta) * alpha) + 
    48 * sqrt(3) * abs2(delta) * real(omega^k * delta)^2 * real(conj(delta) * alpha) - 
    6 * sqrt(3) * abs2(delta)^2 * real(conj(delta) * alpha)) * (8 * real(omega^k * delta)^3 - 4 * abs2(delta) * real(omega^k * delta) - 
    conj(delta)^3) + (sqrt(3) * conj(delta) / abs(delta)^3 * real(conj(delta) * alpha) - conj(alpha) * sqrt(3) / abs(delta)) * 
    (8 * real(omega^k * delta)^3 - 4 * abs2(delta) * real(omega^k * delta) - conj(delta)^3) - 
    4 * sqrt(3) * conj(delta) / abs(delta) * real(omega^k * delta) * real(conj(delta) * alpha))
    
    return dx_A3, dy_A3
end

# At the origin
function analytic_d_A5(delta, alpha, index)
    omega = exp(im * 2 * pi/ 3)
    k = index - 2

    nmz = (64 * real(omega^k * delta)^6 - 48 * abs2(delta) * real(omega^k * delta)^4 - 16 * real(delta^3) * real(omega^k * delta)^3 + 
    12 * abs2(delta)^2 * real(omega^k * delta)^2 + 12 * abs2(delta) * real(delta^3) * real(omega^k * delta) + 3 * abs2(delta)^3)


    dx_A5 = 1/sqrt(nmz) * (-abs(delta)/(2 * nmz) * (-96 * real(omega^k * delta)^4 * real(conj(delta) * alpha) + 
    6 * abs2(delta)^2 * real(conj(delta) * alpha)) * (2 * delta * real(omega^k * delta) + conj(delta)^2) - 
    1/abs(delta) * real(conj(delta) * alpha) * (2 * delta * real(omega^k * delta) + conj(delta)^2) + 
    abs(delta) * (conj(alpha) * conj(delta) - 2 * alpha * real(omega^k * delta)))

    dy_A5 = 1/sqrt(nmz) * (-abs(delta)/(2 * nmz) * (-96 * sqrt(3) * real(omega^k * delta)^4 * real(conj(delta) * alpha) + 
    48 * sqrt(3) * abs2(delta) * real(omega^k * delta)^2 * real(conj(delta) * alpha) - 
    6 * sqrt(3) * abs2(delta)^2 * real(conj(delta) * alpha)) * (2 * delta * real(omega^k * delta) + conj(delta)^2) - 
    sqrt(3)/abs(delta) * real(conj(delta) * alpha) * (2 * delta * real(omega^k * delta) + conj(delta)^2) + 
    abs(delta) * (2 * sqrt(3) * alpha * real(omega^k * delta) - sqrt(3) * conj(alpha) * conj(delta)))
    
    return dx_A5, dy_A5
end

function analytic_og_bc(delta, alpha, index)
    dxA1, dyA1 = analytic_d_A1(delta, alpha, index)
    dxA3, dyA3 = analytic_d_A3(delta, alpha, index)
    dxA5, dyA5 = analytic_d_A5(delta, alpha, index)
    return -2 * imag(conj(dxA1) * dyA1) - 2 * imag(conj(dxA3) * dyA3) - 2 * imag(conj(dxA5) * dyA5)
end

function explicit_og_bc(delta, alpha, index)
    k = index - 2
    if mod(k, 3) == 0
        return -8/(sqrt(3) * (delta^2 + conj(delta)^2 + abs(delta)^2)^2) * (imag(alpha) * imag(delta) + real(alpha) * real(delta)) * 
        (3 * imag(alpha) * real(delta) + imag(delta) * real(alpha))
    elseif mod(k, 3) == 1
        return (2/3) * (real(alpha)^2 / imag(delta)^2 + (-3 * imag(alpha)^2 + 2*sqrt(3) * imag(alpha) * real(alpha) - real(alpha)^2) / 
        (imag(delta) + sqrt(3) * real(delta))^2)
    elseif mod(k, 3) == 2
        return 2*(imag(alpha) * imag(delta) + real(alpha) * real(delta))/(3 * imag(delta)^2 * (imag(delta) - sqrt(3) * real(delta))^3) * 
        (3 * imag(alpha) * imag(delta)^2 - 3 * sqrt(3) * imag(delta) * (imag(alpha) + sqrt(3) * real(alpha)) * real(delta) + 
        sqrt(3) * (2 * imag(delta)^2 + 3 * real(delta)^2) * real(alpha))
    end
end

function bc_no_spinors_analytic(points, spacing, delt, alph, index)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        states = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            val = analytic_eigenvalues(alph, delt, x_new, y_new)[index]
            states[j, :] = analytic_eigenvectors(val, alph, delt, x_new, y_new)
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = dot(states[j, :], states[j + 1, :])
                P *= temp
            else
                temp = dot(states[j, :], states[1, :])
                P *= temp
            end
            if temp != 0
                P /= abs(temp)
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing / sqrt(2), num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing / sqrt(2), num_vertices)
        end
    end
    return berry_list
end



# function analytic_origin_3p(alpha, delta)
#     omega = exp(im * 2 * pi / 3)
#     epsilon = 2 * real(delta)
#     if 2 * real(omega * delta) < epsilon
#         epsilon = 2 * real(omega * delta)
#     end
#     if 2 * real(conj(omega) * delta) < epsilon
#         epsilon = 2 * real(conj(omega) * delta)
#     end
#     A1y = 1/(epsilon^2 + 2 * epsilon * real(delta) + real(delta)^2 - 3 * imag(delta)^2) * 
#     (-1im) * (imag(delta) * real(alpha) + real(delta) * imag(alpha) + imag(alpha) * epsilon)
#     A1x = 1/(epsilon + real(delta)) * (-im * sqrt(3) * imag(delta) * A1y - 1/sqrt(3) * real(alpha))
#     A3x = -1/2 * A1x - sqrt(3)/2 * A1y
#     A3y = sqrt(3)/2 * A1x - 1/2 * A1y
#     A5x = -1/2 * A1x + sqrt(3)/2 * A1y
#     A5y = -sqrt(3)/2 * A1x - 1/2 * A1y
#     bc = -2 * (imag(conj(A1x) * A1y) + imag(conj(A3x) * A3y) + imag(conj(A5x) * A5y))
#     return bc
# end

function g_mu_nu(delta_k, psi_1, psi_2, psi_3)
    gf_psi_1 = gauge_fix(psi_1)
    gf_psi_2 = gauge_fix(psi_2)
    gf_psi_3 = gauge_fix(psi_3)

    dpsi1 = 1/(delta_k) * (gf_psi_2 - gf_psi_1)
    dpsi2 = 1/(delta_k) * (gf_psi_3 - gf_psi_2)

    term1 = dot(dpsi1, dpsi2)
    term2 = dot(dpsi1, gf_psi_2) * dot(gf_psi_2, dpsi2)

    return real(term1 - term2)
end

function patch_g_mu_nu(delta_k, C1, C2, C3, psi_1, psi_2, psi_3)
    gf_psi_1 = Array{ComplexF64}(undef, 3, size(psi_1, 2))
    gf_psi_2 = Array{ComplexF64}(undef, 3, size(psi_2, 2))
    gf_psi_3 = Array{ComplexF64}(undef, 3, size(psi_3, 2))
    for j in 1:3
        gf_psi_1[j, :] = gauge_fix(psi_1[j, :])
        gf_psi_2[j, :] = gauge_fix(psi_2[j, :])
        gf_psi_3[j, :] = gauge_fix(psi_3[j, :])
    end
    # Align gauges
    C1 = C1 / (dot(C2, C1) / abs(dot(C2, C1)))
    C3 = C3 / (dot(C2, C3) / abs(dot(C2, C3)))
    # Compute the <dxCu|dxCu> terms
    term1 = 0
    for j in 1:3
        dpsi1 = 1/delta_k * (C2[j] * (gf_psi_2[j, :] - gf_psi_1[j, :]) + gf_psi_2[j, :] * (C2[j] - C1[j]))
        dpsi2 = 1/delta_k * (C2[j] * (gf_psi_3[j, :] - gf_psi_2[j, :]) + gf_psi_2[j, :] * (C3[j] - C2[j]))
        term1 += dot(dpsi1, dpsi2)
    end
    t2l = 0
    t2r = 0
    term2 = 0
    for j in 1:3
        dpsi1 = 1/delta_k * (C2[j] * (gf_psi_2[j, :] - gf_psi_1[j, :]) + gf_psi_2[j, :] * (C2[j] - C1[j]))
        dpsi2 = 1/delta_k * (C2[j] * (gf_psi_3[j, :] - gf_psi_2[j, :]) + gf_psi_2[j, :] * (C3[j] - C2[j]))
        term2 += dot(dpsi1, C2[j] * gf_psi_2[j, :]) * dot(C2[j] * gf_psi_2[j, :], dpsi2)
        t2l += dot(dpsi1, C2[j] * gf_psi_2[j, :])
        t2r += dot(C2[j] * gf_psi_2[j, :], dpsi2)
    end
    return real(term1 - t2l * t2r)
end

function C_dx(k, alpha, delta, dx)
    kl = k - dx/2 * [1, 0]
    mom = norm(kl)
    theta = polar_angle(kl[1], kl[2])
    ham = H_mft_v2(kl, delta, alpha)
    cl = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    kr = k + dx/2 * [1, 0]
    mom = norm(kr)
    theta = polar_angle(kr[1], kr[2])
    ham = H_mft_v2(kr, delta, alpha)
    cr = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    # Align gauges of cl, cr
    cr = cr / (dot(cl, cr)/abs(dot(cl, cr)))

    return 1/dx * (cr - cl)
end

function C_dy(k, alpha, delta, dy)
    kl = k - dy/2 * [0, 1]
    mom = norm(kl)
    theta = polar_angle(kl[1], kl[2])
    ham = H_mft_v2(kl, delta, alpha)
    cl = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    kr = k + dy/2 * [0, 1]
    mom = norm(kr)
    theta = polar_angle(kr[1], kr[2])
    ham = H_mft_v2(kr, delta, alpha)
    cr = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    
    # Align gauges of cl, cr
    cr = cr / (dot(cl, cr)/abs(dot(cl, cr)))

    return 1/dy * (cr - cl)
end
function ell_k(Delta)
    if angle(Delta) > -2*pi/3 && angle(Delta) <= 0
        return -1
    elseif angle(Delta) > 0 && angle(Delta) <= 2*pi/3
        return 1
    else
        return 0
    end
end
""")

Main.eval("""
function rmg_spinor(Nl, nu, k)
    vec = Array{ComplexF64}(undef, Nl)
    for i in 1:Nl
        vec[i] = nu^(i - 1) * (k[1] + im * k[2])^(i - 1)
    end
    return vec
end

# Berry curvature from just spinor
function rmg_spinor_bc(points, spacing, Nl, nu)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through square plaquette centered at point
        num_vertices = 4
        momenta = Array{Float64}(undef, num_vertices, 2)
        for j in 1:num_vertices
            x_new = points[i, 1] + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = points[i, 2] + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momenta[j, 1] = x_new
            momenta[j, 2] = y_new
        end
        states = Array{ComplexF64}(undef, num_vertices, Nl)
        for j in 1:num_vertices
            sor = normalize(rmg_spinor(Nl, nu, momenta[j, :]))
            states[j, :] = sor
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                P *= dot(states[j, :], states[j + 1, :]) / abs(dot(states[j, :], states[j + 1, :]))
            else
                P *= dot(states[j, :], states[1, :]) / abs(dot(states[j, :], states[1, :]))
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing * sqrt(2), num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing * sqrt(2), num_vertices)
        end
    end
    return berry_list
end

# Berry curvature over all plaquettes
function rmg_patch_bc(points, spacing, Nl, nu, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        spinors = Array{ComplexF64}(undef, num_vertices, 3, Nl)
        grounds = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))
            for m in 1:3
                kappa_pt = [m_kappa * cos(2*pi/3 * (m - 1)), m_kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                sor = normalize(rmg_spinor(Nl, nu, kappa_pt))
                spinors[j, m, :] = gauge_fix(sor)
            end
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = spinor_inner(grounds[j, :], grounds[j + 1, :], spinors[j, :, :], spinors[j + 1, :, :])
                P *= temp / abs(temp)
            else
                temp = spinor_inner(grounds[j, :], grounds[1, :], spinors[j, :, :], spinors[1, :, :])
                P *= temp / abs(temp)
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing * sqrt(2), num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing * sqrt(2), num_vertices)
        end
    end
    return berry_list
end

function rmg_weighted_sum_bc(points, spacing, Nl, nu, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        x0 = points[i, 1]
        y0 = points[i, 2]
        momentum = norm([x0, y0])
        theta = polar_angle(x0, y0)
        ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
        grounds = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))
        bc_tot = 0
        for j in 1:3
            t_pt = m_kappa * [cos(2*pi/3 * (j - 1)), sin(2*pi/3 * (j - 1))] + [x0, y0]
            pt = reshape(t_pt, (1, 2))
            spin_bc = rmg_spinor_bc(pt, spacing, Nl, nu)[1]
            bc_tot += abs(grounds[j])^2 * spin_bc
        end
        berry_list[i] = bc_tot
    end
    return berry_list
end

function rmg_equal_weighted_sum_bc(points, spacing, Nl, nu, m_kappa)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        x0 = points[i, 1]
        y0 = points[i, 2]
        bc_tot = 0
        for j in 1:3
            t_pt = m_kappa * [cos(2*pi/3 * (j - 1)), sin(2*pi/3 * (j - 1))] + [x0, y0]
            pt = reshape(t_pt, (1, 2))
            spin_bc = rmg_spinor_bc(pt, spacing, Nl, nu)[1]
            bc_tot += 1/3 * spin_bc
        end
        berry_list[i] = bc_tot
    end
    return berry_list
end

function rmg_decoupled_bc(points, spacing, Nl, nu, m_kappa, vF, delt, alph)
    return rmg_weighted_sum_bc(points, spacing, Nl, nu, m_kappa, vF, delt, alph) .+ bc_no_spinors(points, spacing, vF, delt, alph)
end

function rmg_gxx(k, spacing, Nl, nu)
    kl = k - spacing * [1, 0]
    psi_l = normalize(rmg_spinor(Nl, nu, kl))

    psi_0 = normalize(rmg_spinor(Nl, nu, k))
    
    kr = k + spacing * [1, 0]
    psi_r = normalize(rmg_spinor(Nl, nu, kr))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function rmg_gyy(k, spacing, Nl, nu)
    kl = k - spacing * [0, 1]
    psi_l = normalize(rmg_spinor(Nl, nu, kl))

    psi_0 = normalize(rmg_spinor(Nl, nu, k))
    
    kr = k + spacing * [0, 1]
    psi_r = normalize(rmg_spinor(Nl, nu, kr))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function rmg_gxy(k, spacing, Nl, nu)
    kl = k - spacing * [1, 1]
    psi_l = normalize(rmg_spinor(Nl, nu, kl))

    psi_0 = normalize(rmg_spinor(Nl, nu, k))
    
    kr = k + spacing * [1, 1]
    psi_r = normalize(rmg_spinor(Nl, nu, kr))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end
# quantum metric is symmetric
function rmg_gyx(k, spacing, Nl, nu)
    kl = k - spacing * [1, -1]
    psi_l = normalize(rmg_spinor(Nl, nu, kl))

    psi_0 = normalize(rmg_spinor(Nl, nu, k))
    
    kr = k + spacing * [1, -1]
    psi_r = normalize(rmg_spinor(Nl, nu, kr))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function rmg_quantum_metric(points, spacing, Nl, nu)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = rmg_gxx(points[i, :], spacing, Nl, nu)
        metric[2, i] = rmg_gxy(points[i, :], spacing, Nl, nu)
        metric[3, i] = rmg_gyx(points[i, :], spacing, Nl, nu)
        metric[4, i] = rmg_gyy(points[i, :], spacing, Nl, nu)
    end
    return metric
end

function rmg_3p_gxx(k, spacing, Nl, nu, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa_pts[1, :] = m_kappa * [1, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)]

    spinors = Array{ComplexF64}(undef, 3, 3, 2*Nl)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [1, 0]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(rmg_spinor(Nl, nu, kl + kappa_pts[j, :]))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(rmg_spinor(Nl, nu, k + kappa_pts[j, :]))
    end
    
    kr = k + spacing * [1, 0]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(rmg_spinor(Nl, nu, kr + kappa_pts[j, :]))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end

function rmg_3p_gyy(k, spacing, Nl, nu, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa_pts[1, :] = m_kappa * [1, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)]

    spinors = Array{ComplexF64}(undef, 3, 3, 2*Nl)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [0, 1]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(rmg_spinor(Nl, nu, kl + kappa_pts[j, :]))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(rmg_spinor(Nl, nu, k + kappa_pts[j, :]))
    end
    
    kr = k + spacing * [0, 1]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(rmg_spinor(Nl, nu, kr + kappa_pts[j, :]))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end

function rmg_patch_qm(points, spacing, Nl, nu, m_kappa, vF, delt, alph)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = rmg_3p_gxx(points[i, :], spacing, Nl, nu, m_kappa, vF, delt, alph)
        metric[4, i] = rmg_3p_gyy(points[i, :], spacing, Nl, nu, m_kappa, vF, delt, alph)
    end
    return metric
end

function rmg_epsilon_K(k, nu)
    return (real(sum(abs2, k)))^(3/2)
end

function rmg_ff(Nl, nu, k1, k2)
    psi1 = normalize(rmg_spinor(Nl, nu, k1))
    psi2 = normalize(rmg_spinor(Nl, nu, k2))
    return dot(psi1, psi2)
end

# linear potential overlaps
function rmg_ff_linear(Nl, nu, k1, k2)
    psi1 = normalize(rmg_spinor(Nl, nu, k1))
    psi2 = normalize(rmg_spinor(Nl, nu, k2))
    val = 0
    for j in 1:Nl
        val += (j - 1) * conj(psi1[j]) * psi2[j]
    end
    return val
end

function rmg_full_mBZ_ham(k, Nl, nu, kappa, V)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    ham = zeros(ComplexF64, 7, 7)

    # diagonal matrix elements
    ham[1, 1] = rmg_epsilon_K(k, nu)
    ham[2, 2] = rmg_epsilon_K(k .+ g1, nu)
    ham[3, 3] = rmg_epsilon_K(k .+ g2, nu)
    ham[4, 4] = rmg_epsilon_K(k .+ g2 .- g1, nu)
    ham[5, 5] = rmg_epsilon_K(k .- g1, nu)
    ham[6, 6] = rmg_epsilon_K(k .- g2, nu)
    ham[7, 7] = rmg_epsilon_K(k .+ g1 .- g2, nu)

    # off-diagonal matrix elements
    ham[1, 2] = V/2 * rmg_ff(Nl, nu, k, k .+ g1)
    ham[1, 3] = V/2 * rmg_ff(Nl, nu, k, k .+ g2)
    ham[1, 4] = V/2 * rmg_ff(Nl, nu, k, k .+ g2 .- g1)
    ham[1, 5] = V/2 * rmg_ff(Nl, nu, k, k .- g1)
    ham[1, 6] = V/2 * rmg_ff(Nl, nu, k, k .- g2)
    ham[1, 7] = V/2 * rmg_ff(Nl, nu, k, k .+ g1 .- g2)

    ham[2, 1] = V/2 * rmg_ff(Nl, nu, k .+ g1, k)
    ham[2, 3] = V/2 * rmg_ff(Nl, nu, k .+ g1, k .+ g2)
    ham[2, 7] = V/2 * rmg_ff(Nl, nu, k .+ g1, k .+ g1 .- g2)

    ham[3, 1] = V/2 * rmg_ff(Nl, nu, k .+ g2, k)
    ham[3, 2] = V/2 * rmg_ff(Nl, nu, k .+ g2, k .+ g1)
    ham[3, 4] = V/2 * rmg_ff(Nl, nu, k .+ g2, k .+ g2 .- g1)

    ham[4, 1] = V/2 * rmg_ff(Nl, nu, k .+ g2 .- g1, k)
    ham[4, 3] = V/2 * rmg_ff(Nl, nu, k .+ g2 .- g1, k .+ g2)
    ham[4, 5] = V/2 * rmg_ff(Nl, nu, k .+ g2 .- g1, k .- g1)

    ham[5, 1] = V/2 * rmg_ff(Nl, nu, k .- g1, k)
    ham[5, 4] = V/2 * rmg_ff(Nl, nu, k .- g1, k .+ g1 .- g1)
    ham[5, 6] = V/2 * rmg_ff(Nl, nu, k .- g1, k .- g2)

    ham[6, 1] = V/2 * rmg_ff(Nl, nu, k .- g2, k)
    ham[6, 5] = V/2 * rmg_ff(Nl, nu, k .- g2, k .- g1)
    ham[6, 7] = V/2 * rmg_ff(Nl, nu, k .- g2, k .+ g1 .- g2)

    ham[7, 1] = V/2 * rmg_ff(Nl, nu, k .+ g1 .- g2, k)
    ham[7, 2] = V/2 * rmg_ff(Nl, nu, k .+ g1 .- g2, k .+ g1)
    ham[7, 6] = V/2 * rmg_ff(Nl, nu, k .+ g1 .- g2, k .- g2)

    return ham
end


# Berry curvature over all plaquettes
function rmg_full_mBZ_bc(points, spacing, Nl, nu, kappa, V)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    mom_list = [[0, 0], g1, g2, g2 .- g1, -g1, -g2, g1 .- g2]
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        spinors = Array{ComplexF64}(undef, num_vertices, 7, Nl)
        grounds = Array{ComplexF64}(undef, num_vertices, 7)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            ham = rmg_full_mBZ_ham([x_new, y_new], Nl, nu, kappa, V)
            grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))
            for m in 1:7
                momentum = mom_list[m] .+ [x_new, y_new]
                sor = normalize(rmg_spinor(Nl, nu, momentum))
                spinors[j, m, :] = gauge_fix(sor)
            end
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = spinor_inner(grounds[j, :], grounds[j + 1, :], spinors[j, :, :], spinors[j + 1, :, :])
                P *= temp / abs(temp)
            else
                temp = spinor_inner(grounds[j, :], grounds[1, :], spinors[j, :, :], spinors[1, :, :])
                P *= temp / abs(temp)
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing * sqrt(2), num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing * sqrt(2), num_vertices)
        end
    end
    return berry_list
end


function rmg_full_mBZ_ham_linear(k, Nl, nu, kappa, V)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    ham = zeros(ComplexF64, 7, 7)

    # diagonal matrix elements
    ham[1, 1] = rmg_epsilon_K(k, nu)
    ham[2, 2] = rmg_epsilon_K(k .+ g1, nu)
    ham[3, 3] = rmg_epsilon_K(k .+ g2, nu)
    ham[4, 4] = rmg_epsilon_K(k .+ g2 .- g1, nu)
    ham[5, 5] = rmg_epsilon_K(k .- g1, nu)
    ham[6, 6] = rmg_epsilon_K(k .- g2, nu)
    ham[7, 7] = rmg_epsilon_K(k .+ g1 .- g2, nu)

    # off-diagonal matrix elements
    ham[1, 2] = V/2 * rmg_ff_linear(Nl, nu, k, k .+ g1)
    ham[1, 3] = V/2 * rmg_ff_linear(Nl, nu, k, k .+ g2)
    ham[1, 4] = V/2 * rmg_ff_linear(Nl, nu, k, k .+ g2 .- g1)
    ham[1, 5] = V/2 * rmg_ff_linear(Nl, nu, k, k .- g1)
    ham[1, 6] = V/2 * rmg_ff_linear(Nl, nu, k, k .- g2)
    ham[1, 7] = V/2 * rmg_ff_linear(Nl, nu, k, k .+ g1 .- g2)

    ham[2, 1] = V/2 * rmg_ff_linear(Nl, nu, k .+ g1, k)
    ham[2, 3] = V/2 * rmg_ff_linear(Nl, nu, k .+ g1, k .+ g2)
    ham[2, 7] = V/2 * rmg_ff_linear(Nl, nu, k .+ g1, k .+ g1 .- g2)

    ham[3, 1] = V/2 * rmg_ff_linear(Nl, nu, k .+ g2, k)
    ham[3, 2] = V/2 * rmg_ff_linear(Nl, nu, k .+ g2, k .+ g1)
    ham[3, 4] = V/2 * rmg_ff_linear(Nl, nu, k .+ g2, k .+ g2 .- g1)

    ham[4, 1] = V/2 * rmg_ff_linear(Nl, nu, k .+ g2 .- g1, k)
    ham[4, 3] = V/2 * rmg_ff_linear(Nl, nu, k .+ g2 .- g1, k .+ g2)
    ham[4, 5] = V/2 * rmg_ff_linear(Nl, nu, k .+ g2 .- g1, k .- g1)

    ham[5, 1] = V/2 * rmg_ff_linear(Nl, nu, k .- g1, k)
    ham[5, 4] = V/2 * rmg_ff_linear(Nl, nu, k .- g1, k .+ g1 .- g1)
    ham[5, 6] = V/2 * rmg_ff_linear(Nl, nu, k .- g1, k .- g2)

    ham[6, 1] = V/2 * rmg_ff_linear(Nl, nu, k .- g2, k)
    ham[6, 5] = V/2 * rmg_ff_linear(Nl, nu, k .- g2, k .- g1)
    ham[6, 7] = V/2 * rmg_ff_linear(Nl, nu, k .- g2, k .+ g1 .- g2)

    ham[7, 1] = V/2 * rmg_ff_linear(Nl, nu, k .+ g1 .- g2, k)
    ham[7, 2] = V/2 * rmg_ff_linear(Nl, nu, k .+ g1 .- g2, k .+ g1)
    ham[7, 6] = V/2 * rmg_ff_linear(Nl, nu, k .+ g1 .- g2, k .- g2)

    return ham
end


# Berry curvature over all plaquettes
function rmg_full_mBZ_bc_linear(points, spacing, Nl, nu, kappa, V)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    mom_list = [[0, 0], g1, g2, g2 .- g1, -g1, -g2, g1 .- g2]
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        spinors = Array{ComplexF64}(undef, num_vertices, 7, Nl)
        grounds = Array{ComplexF64}(undef, num_vertices, 7)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            ham = rmg_full_mBZ_ham_linear([x_new, y_new], Nl, nu, kappa, V)
            grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))
            for m in 1:7
                momentum = mom_list[m] .+ [x_new, y_new]
                sor = normalize(rmg_spinor(Nl, nu, momentum))
                spinors[j, m, :] = gauge_fix(sor)
            end
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = spinor_inner(grounds[j, :], grounds[j + 1, :], spinors[j, :, :], spinors[j + 1, :, :])
                P *= temp / abs(temp)
            else
                temp = spinor_inner(grounds[j, :], grounds[1, :], spinors[j, :, :], spinors[1, :, :])
                P *= temp / abs(temp)
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing * sqrt(2), num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing * sqrt(2), num_vertices)
        end
    end
    return berry_list
end


function num_mBZ(shells)
    return 1 + 3 * shells * (shells + 1)
end


# calculates all possible combinations of g1 and g2 that land within the mBZ shells
function sgn_partitions(n)
    pairs = []
    for a in -n:n
        for b in -n:n
            if -n <= a + b <= n
                push!(pairs, [a, b])
            end
        end
    end
    return pairs
end

function is_valid_pair(n, part)
    a, b = part
    return -n <= a <= n && -n <= b <= n && -n <= a + b <= n
end

function rmg_ham_mBZ_shells!(g1, g2, gi, gj, partitions, ham, k, Nl, nu, kappa, V, shells)
    mBZ_count = num_mBZ(shells)
    for i in 1:mBZ_count
        part_i = partitions[i]
        for j in i:mBZ_count
            part_j = partitions[j]
            if i != j && is_valid_pair(shells, part_i - part_j)
                gi = part_i[1] * g1 + part_i[2] * g2
                gj = part_j[1] * g1 + part_j[2] * g2
                ham[i, j] = V/2 * rmg_ff(Nl, nu, k .+ gi, k .+ gj)
                ham[j, i] = V/2 * rmg_ff(Nl, nu, k .+ gj, k .+ gi)
            end
            if i == j
                ham[j, j] = rmg_epsilon_K(part_j[1] * g1 + part_j[2] * g2 .+ k, nu)
            end
        end
    end
    return ham
end

function rmg_ham_mBZ_shells(k, Nl, nu, kappa, V, shells)
    mBZ_count = num_mBZ(shells)
    ham = zeros(ComplexF64, mBZ_count, mBZ_count)
    partitions = sgn_partitions(shells)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    gi = zeros(2)
    gj = zeros(2)
    rmg_ham_mBZ_shells!(g1, g2, gi, gj, partitions, ham, k, Nl, nu, kappa, V, shells)
end

function rmg_ham_mBZ_shells_linear!(g1, g2, gi, gj, partitions, ham, k, Nl, nu, kappa, V, shells)
    mBZ_count = num_mBZ(shells)
    for i in 1:mBZ_count
        part_i = partitions[i]
        for j in i:mBZ_count
            part_j = partitions[j]
            if i != j && is_valid_pair(shells, part_i - part_j)
                gi = part_i[1] * g1 + part_i[2] * g2
                gj = part_j[1] * g1 + part_j[2] * g2
                ham[i, j] = V/2 * rmg_ff_linear(Nl, nu, k .+ gi, k .+ gj)
                ham[j, i] = V/2 * rmg_ff_linear(Nl, nu, k .+ gj, k .+ gi)
            end
            if i == j
                ham[j, j] = rmg_epsilon_K(part_j[1] * g1 + part_j[2] * g2 .+ k, nu)
            end
        end
    end
    return ham
end

function rmg_ham_mBZ_shells_linear(k, Nl, nu, kappa, V, shells)
    mBZ_count = num_mBZ(shells)
    ham = zeros(ComplexF64, mBZ_count, mBZ_count)
    partitions = sgn_partitions(shells)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    gi = zeros(2)
    gj = zeros(2)
    rmg_ham_mBZ_shells_linear!(g1, g2, gi, gj, partitions, ham, k, Nl, nu, kappa, V, shells)
end

function partitions_to_momenta(shells, kappa)
    partitions = sgn_partitions(shells)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    momenta_list = Array{Float64}(undef, length(partitions), 2)
    for j in 1:length(partitions)
        part = partitions[j]
        momenta_list[j, :] = part[1] * g1 + part[2] * g2
    end
    return momenta_list
end

function uniform_mBZ_bc!(g1, g2, gi, gj, num_vertices, plaq_area, mom_list, partitions, ham, spinors, grounds, k, Nl, nu, kappa, V, shells, index, spacing)
    for j in 1:num_vertices
        x_new = k[1] + spacing/2 * cos(2 * pi * (j - 1) / num_vertices)
        y_new = k[2] + spacing/2 * sin(2 * pi * (j - 1) / num_vertices)
        ham = rmg_ham_mBZ_shells!(g1, g2, gi, gj, partitions, ham, [x_new, y_new], Nl, nu, kappa, V, shells)
        grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))
        for m in 1:size(mom_list, 1)
            spinors[j, m, :] = normalize(rmg_spinor(Nl, nu, mom_list[m, :] .+ [x_new, y_new]))
        end
    end
    P = 1
    for j in 1:num_vertices
        if j < num_vertices
            temp = spinor_inner(grounds[j, :], grounds[j + 1, :], spinors[j, :, :], spinors[j + 1, :, :])
            P *= temp / abs(temp)
        else
            temp = spinor_inner(grounds[j, :], grounds[1, :], spinors[j, :, :], spinors[1, :, :])
            P *= temp / abs(temp)
        end
    end
    if abs(imag(P)) < 10^(-16)
        return -(angle(real(P))) / plaq_area
    else
        return -angle(P) / plaq_area
    end
end

function uniform_mBZ_bc(k, Nl, nu, kappa, V, shells, index, spacing)
    num_vertices = 4
    mom_list = partitions_to_momenta(shells, kappa)

    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]

    gi = zeros(2)
    gj = zeros(2)

    mBZ_count = num_mBZ(shells)
    partitions = sgn_partitions(shells)
    ham = zeros(ComplexF64, mBZ_count, mBZ_count)
    spinors = Array{ComplexF64}(undef, num_vertices, size(mom_list, 1), Nl)
    grounds = Array{ComplexF64}(undef, num_vertices, size(mom_list, 1))
    plaq_area = area(spacing * sqrt(2), num_vertices)

    uniform_mBZ_bc!(g1, g2, gi, gj, num_vertices, plaq_area, mom_list, partitions, ham, spinors, grounds, k, Nl, nu, kappa, V, shells, index, spacing)
end

function linear_mBZ_bc!(g1, g2, gi, gj, num_vertices, plaq_area, mom_list, partitions, ham, spinors, grounds, k, Nl, nu, kappa, V, shells, index, spacing)
    for j in 1:num_vertices
        x_new = k[1] + spacing/2 * cos(2 * pi * (j - 1) / num_vertices)
        y_new = k[2] + spacing/2 * sin(2 * pi * (j - 1) / num_vertices)
        ham = rmg_ham_mBZ_shells_linear!(g1, g2, gi, gj, partitions, ham, [x_new, y_new], Nl, nu, kappa, V, shells)
        grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))
        for m in 1:size(mom_list, 1)
            spinors[j, m, :] = normalize(rmg_spinor(Nl, nu, mom_list[m, :] .+ [x_new, y_new]))
        end
    end
    P = 1
    for j in 1:num_vertices
        if j < num_vertices
            temp = spinor_inner(grounds[j, :], grounds[j + 1, :], spinors[j, :, :], spinors[j + 1, :, :])
            P *= temp / abs(temp)
        else
            temp = spinor_inner(grounds[j, :], grounds[1, :], spinors[j, :, :], spinors[1, :, :])
            P *= temp / abs(temp)
        end
    end
    if abs(imag(P)) < 10^(-16)
        return -(angle(real(P))) / plaq_area
    else
        return -angle(P) / plaq_area
    end
end

function linear_mBZ_bc(k, Nl, nu, kappa, V, shells, index, spacing)
    num_vertices = 4
    mom_list = partitions_to_momenta(shells, kappa)

    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]

    gi = zeros(2)
    gj = zeros(2)

    mBZ_count = num_mBZ(shells)
    partitions = sgn_partitions(shells)
    ham = zeros(ComplexF64, mBZ_count, mBZ_count)
    spinors = Array{ComplexF64}(undef, num_vertices, size(mom_list, 1), Nl)
    grounds = Array{ComplexF64}(undef, num_vertices, size(mom_list, 1))
    plaq_area = area(spacing * sqrt(2), num_vertices)

    linear_mBZ_bc!(g1, g2, gi, gj, num_vertices, plaq_area, mom_list, partitions, ham, spinors, grounds, k, Nl, nu, kappa, V, shells, index, spacing)
end
""")

Main.eval("""
function uniform_bc_point(k, Nl, nu, kappa, V, shells, index, spacing)
    num_vertices = 4
    mom_list = partitions_to_momenta(shells, kappa)

    mBZ_count = num_mBZ(shells)
    partitions = sgn_partitions(shells)
    ham = zeros(ComplexF64, mBZ_count, mBZ_count)
    spinors = Array{ComplexF64}(undef, num_vertices, size(mom_list, 1), Nl)
    grounds = Array{ComplexF64}(undef, num_vertices, size(mom_list, 1))
    plaq_area = area(spacing * sqrt(2), num_vertices)

    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]

    gi = zeros(2)
    gj = zeros(2)
    
    point = reshape(k, (1, 2))

    berry = uniform_mBZ_bc!(g1, g2, gi, gj, num_vertices, plaq_area, mom_list, partitions, ham, spinors, grounds, point, Nl, nu, kappa, V, shells, index, spacing)

    return berry
end
""")

if __name__ == "__main__":

    kx = float(sys.argv[1])
    ky = float(sys.argv[2])
    Nl = int(sys.argv[3])
    nu = float(sys.argv[4])
    kappa = float(sys.argv[5])
    V = float(sys.argv[6])
    shells = int(sys.argv[7])
    index = int(sys.argv[8])
    spacing = float(sys.argv[9])
    dir_name = sys.argv[10]

    bc = Main.uniform_bc_point([kx, ky], Nl, nu, kappa, V, shells, index, spacing)

    outfilename = "./" + dir_name + "/kx="+str(kx)+"_ky="+str(ky)+".pkl"   
    results={'bc': bc, 'kx': kx, 'ky': ky, 'Nl': Nl, 'nu': nu, 'kappa': kappa, 'V': V, 'shells': shells, 'index': index, 'spacing': spacing}
    new_file=open(outfilename,"wb")
    pickle.dump(results, new_file)