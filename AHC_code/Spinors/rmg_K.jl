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