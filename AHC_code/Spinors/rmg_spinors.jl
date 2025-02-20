function r_f(k, a)
    kx = k[1]
    ky = k[2]
    return exp(1im * ky * a / sqrt(3)) * (1 + 2 * exp(-1im * 3 * ky * a / (2 * sqrt(3))) * cos(kx * a / 2))
end

function uAB(uD, Nl, l)
    return uD * (l + 1 - (1/2) * (Nl - 1))
end

function h0(uD, Nl, l, t0, a, k)
    uA = uAB(uD, Nl, l)
    uB = uAB(uD, Nl, l)
    h0 = zeros(ComplexF64, 2, 2)
    h0[1, 1] = uA
    h0[2, 2] = uB
    h0[1, 2] = -t0 * r_f(k, a)
    h0[2, 1] = -t0 * conj(r_f(k, a))
    return h0
end

function h1(t1, t3, t4, a, k)
    h1 = zeros(ComplexF64, 2, 2)
    h1[1, 1] = t4 * r_f(k, a)
    h1[1, 2] = t3 * conj(r_f(k, a))
    h1[2, 1] = t1
    h1[2, 2] = t4 * r_f(k, a)
    return h1
end

function h2(t2)
    h2 = zeros(ComplexF64, 2, 2)
    h2[1,2] = t2/2
    return h2
end

function hRG(a, uD, t0, t1, t2, t3, t4, Nl, k)
    # calculate the portion of hRG that has h0 on the diagonal
    ham0 = zeros(2 * Nl, 2 * Nl)
    for l in 1:Nl
        temp = zeros(Nl, Nl)
        temp[l, l] = 1
        ham0 += kron(h0(uD, Nl, l, t0, a, k), temp)
    end
    # calculate the portion of hRG that has h1 1 above and below the diagonal
    utri1 = zeros(Nl, Nl)
    for i in 2:Nl
        utri1[i - 1, i] = 1
    end
    ham1 = kron(h1(t1, t3, t4, a, k), utri1) + adjoint(kron(h1(t1, t3, t4, a, k), utri1))
    # calculate the portion of hRG that h2 2 above and below the diagonal
    utri2 = zeros(Nl, Nl)
    for i in 3:Nl
        utri2[i - 2, i] = 1
    end
    ham2 = kron(h2(t2), utri2) + adjoint(kron(h2(t2), utri2))
    # get all the contributions of hRG from the various matrices in sublattice space
    return ham0 + ham1 + ham2
end

function rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k, index)
    ham = hRG(a, uD, t0, t1, t2, t3, t4, Nl, k)
    return eigvecs(Hermitian(ham))[:, index]
end

# Berry curvature from just spinor
function rmg_spinor_bc(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
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
        states = Array{ComplexF64}(undef, num_vertices, 2*Nl)
        for j in 1:num_vertices
            sor = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, momenta[j, :], index))
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
            berry_list[i] = -(angle(real(P))) / area(spacing, num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing, num_vertices)
        end
    end
    return berry_list
end

# Berry curvature over all plaquettes
function rmg_patch_bc(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    kappa = 4*pi/(3*a)
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        spinors = Array{ComplexF64}(undef, num_vertices, 3, 2*Nl)
        grounds = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            grounds[j, :] = normalize(eigvecs(Hermitian(ham))[:, 1])
            for m in 1:3
                kappa_pt = [kappa, 0] + [m_kappa * cos(2*pi/3 * (m - 1)), m_kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                sor = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kappa_pt, index))
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
            berry_list[i] = -(angle(real(P))) / area(spacing, num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing, num_vertices)
        end
    end
    return berry_list
end

function rmg_weighted_sum_bc(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    kappa = 4*pi/(3*a)
    for i in 1:size(points)[1]
        x0 = points[i, 1]
        y0 = points[i, 2]
        momentum = norm([x0, y0])
        theta = polar_angle(x0, y0)
        ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
        grounds = normalize(eigvecs(Hermitian(ham))[:, 1])
        bc_tot = 0
        for j in 1:3
            t_pt = [kappa, 0] + m_kappa * [cos(2*pi/3 * (j - 1)), sin(2*pi/3 * (j - 1))] + [x0, y0]
            pt = reshape(t_pt, (1, 2))
            spin_bc = rmg_spinor_bc(pt, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)[1]
            bc_tot += abs(grounds[j])^2 * spin_bc
        end
        berry_list[i] = bc_tot
    end
    return berry_list
end

function rmg_decoupled_bc(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
    return rmg_weighted_sum_bc(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph) .+ bc_no_spinors(points, spacing, vF, delt, alph)
end

function rmg_gxx(k, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
    kl = k - spacing * [1, 0]
    psi_l = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kl, index))

    psi_0 = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k, index))
    
    kr = k + spacing * [1, 0]
    psi_r = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kr, index))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function rmg_gyy(k, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
    kl = k - spacing * [0, 1]
    psi_l = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kl, index))

    psi_0 = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k, index))
    
    kr = k + spacing * [0, 1]
    psi_r = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kr, index))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function rmg_gxy(k, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
    kl = k - spacing * [1, 1]
    psi_l = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kl, index))

    psi_0 = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k, index))
    
    kr = k + spacing * [1, 1]
    psi_r = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kr, index))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end
# quantum metric is symmetric
function rmg_gyx(k, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
    kl = k - spacing * [1, -1]
    psi_l = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kl, index))

    psi_0 = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k, index))
    
    kr = k + spacing * [1, -1]
    psi_r = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kr, index))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function rmg_quantum_metric(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = rmg_gxx(points[i, :], spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
        metric[2, i] = rmg_gxy(points[i, :], spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
        metric[3, i] = rmg_gyx(points[i, :], spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
        metric[4, i] = rmg_gyy(points[i, :], spacing, a, uD, t0, t1, t2, t3, t4, Nl, index)
    end
    return metric
end

function rmg_3p_gxx(k, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa = 4*pi/(3*a)
    kappa_pts[1, :] = m_kappa * [1, 0] + [kappa, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)] + [kappa, 0]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)] + [kappa, 0]

    spinors = Array{ComplexF64}(undef, 3, 3, 2*Nl)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [1, 0]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kl + kappa_pts[j, :], index))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k + kappa_pts[j, :], index))
    end
    
    kr = k + spacing * [1, 0]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kr + kappa_pts[j, :], index))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end

function rmg_3p_gyy(k, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa = 4*pi/(3*a)
    kappa_pts[1, :] = m_kappa * [1, 0] + [kappa, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)] + [kappa, 0]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)] + [kappa, 0]

    spinors = Array{ComplexF64}(undef, 3, 3, 2*Nl)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [0, 1]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kl + kappa_pts[j, :], index))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k + kappa_pts[j, :], index))
    end
    
    kr = k + spacing * [0, 1]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, kr + kappa_pts[j, :], index))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end

function rmg_patch_qm(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = rmg_3p_gxx(points[i, :], spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
        metric[4, i] = rmg_3p_gyy(points[i, :], spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, m_kappa, vF, delt, alph)
    end
    return metric
end