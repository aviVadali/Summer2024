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
    for m in 1:3
        val += conj(C1[m]) * C2[m] * dot(q1_spinor[m, :], q2_spinor[m, :])
    end
    norm1 = 0
    norm2 = 0
    for m in 1:3
        norm1 += conj(C1[m]) * C1[m] * dot(q1_spinor[m, :], q1_spinor[m, :])
        norm2 += conj(C2[m]) * C2[m] * dot(q2_spinor[m, :], q2_spinor[m, :])
    end
    return val / sqrt(norm1 * norm2)
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
    epsilons[3] = real(2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 0*2 * pi/3))
    epsilons[2] = real(2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 1*2 * pi/3))
    epsilons[1] = real(2 * sqrt(B/3) * cos(1/3 * acos(3*C / (2 * B) * sqrt(3/B)) - 2*2 * pi/3))
    return epsilons
end

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