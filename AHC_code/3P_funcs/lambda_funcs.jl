function lambda_spinor(lambda, k)
    kx, ky = k
    vec = Array{ComplexF64}(undef, 2)
    nmz = sqrt(1 + lambda^2 * (kx^2 + ky^2))
    vec[1] = 1/nmz
    vec[2] = lambda * (kx + im * ky)/nmz
    return vec
end

function lambda_spinor_bc(points, spacing, lambda)
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
        states = Array{ComplexF64}(undef, num_vertices, 2)
        for j in 1:num_vertices
            sor = lambda_spinor(lambda, momenta[j, :])
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

function anal_lambda_spinor_bc(lambda, k)
    kx, ky = k
    return -2 * lambda^2 / (1 + lambda^2 * (kx^2 + ky^2))^2
end

# Berry curvature over all plaquettes
function lambda_patch_bc(points, spacing, lambda, kappa, vF, delt, alph, index)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        spinors = Array{ComplexF64}(undef, num_vertices, 3, 2)
        grounds = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            ham = H_mft_v2([x_new, y_new], delt, alph) + H_k_v2([x_new, y_new], vF)
            grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, index]))
            for m in 1:3
                kappa_pt = [kappa * cos(2*pi/3 * (m - 1)), kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                sor = lambda_spinor(lambda, kappa_pt)
                spinors[j, m, :] = sor
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


function lambda_delta_symm(kappa, lambda)
    omega = exp(im * 2 * pi/3)
    delta = (1 + omega * lambda^2 * kappa^2)/(1 + lambda^2 * kappa^2)
    return delta
end

function lambda_alpha_symm(kappa, lambda)
    omega = exp(im * 2 * pi/3)
    alpha = -im * sqrt(3) * kappa * lambda^2 * (1 - omega * kappa^2 * lambda^2)/(1 + kappa^2 * lambda^2)^2
    return alpha
end

function lambda_delta_asymm(kappa, lambda)
    omega = exp(im * 2 * pi/3)
    delta = (1 - omega * lambda^2 * kappa^2)/(1 + lambda^2 * kappa^2)
    return delta
end

function lambda_alpha_asymm(kappa, lambda)
    omega = exp(im * 2 * pi/3)
    alpha = kappa * lambda^2/(1 + kappa^2 * lambda^2)^2 * (2 + im * sqrt(3) * (1 - omega * lambda^2 * kappa^2))
    return alpha
end

function lambda_delta(kappa, lambda, v1, v2)
    omega = exp(im * 2 * pi/3)
    delta = (v1 + v2 * omega * lambda^2 * kappa^2)/(1 + lambda^2 * kappa^2)
    return delta
end

function lambda_alpha(kappa, lambda, v1, v2)
    omega = exp(im * 2 * pi/3)
    phase = exp(im * pi/6)
    alpha = (kappa * lambda^2 / (2 * (1 + lambda^2 * kappa^2)^2)) * (2*v1 + v2 * (4*conj(omega) - 2*sqrt(3)*phase*kappa^2*lambda^2))
    return alpha
end

function lambda_parent(lambda, v1, v2, kappa, vF, q)
    delta = lambda_delta(kappa, lambda, v1, v2)
    alpha = lambda_alpha(kappa, lambda, v1, v2)

    kappa1 = kappa * [1, 0]
    kappa3 = kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa5 = kappa * [cos(4*pi/3), sin(4*pi/3)]

    ham = H_k_v2(q, vF) + H_mft_v2(q, delta, alpha)

    t_vec = eigvecs(Hermitian(ham))[:, 1]

    vec = gauge_fix(normalize(t_vec))

    bc1 = anal_lambda_spinor_bc(lambda, kappa1 .+ q)
    bc3 = anal_lambda_spinor_bc(lambda, kappa3 .+ q)
    bc5 = anal_lambda_spinor_bc(lambda, kappa5 .+ q)

    return abs2(vec[1]) * bc1 + abs2(vec[2]) * bc3 + abs2(vec[3]) * bc5
end

function lambda_symm_parent(lambda, kappa, vF, q)
    delta = -lambda_delta_symm(kappa, lambda)
    alpha = -lambda_alpha_symm(kappa, lambda)

    kappa1 = kappa * [1, 0]
    kappa3 = kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa5 = kappa * [cos(4*pi/3), sin(4*pi/3)]

    ham = H_k_v2(q, vF) + H_mft_v2(q, delta, alpha)

    t_vec = eigvecs(Hermitian(ham))[:, 1]

    vec = gauge_fix(normalize(t_vec))

    bc1 = anal_lambda_spinor_bc(lambda, kappa1 .+ q)
    bc3 = anal_lambda_spinor_bc(lambda, kappa3 .+ q)
    bc5 = anal_lambda_spinor_bc(lambda, kappa5 .+ q)

    return abs2(vec[1]) * bc1 + abs2(vec[2]) * bc3 + abs2(vec[3]) * bc5
end

function lambda_asymm_parent(lambda, kappa, vF, q)
    delta = lambda_delta_asymm(kappa, lambda)
    alpha = lambda_alpha_asymm(kappa, lambda)

    kappa1 = kappa * [1, 0]
    kappa3 = kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa5 = kappa * [cos(4*pi/3), sin(4*pi/3)]

    ham = H_k_v2(q, vF) + H_mft_v2(q, delta, alpha)

    t_vec = eigvecs(Hermitian(ham))[:, 1]

    vec = gauge_fix(normalize(t_vec))

    bc1 = anal_lambda_spinor_bc(lambda, kappa1 .+ q)
    bc3 = anal_lambda_spinor_bc(lambda, kappa3 .+ q)
    bc5 = anal_lambda_spinor_bc(lambda, kappa5 .+ q)

    return abs2(vec[1]) * bc1 + abs2(vec[2]) * bc3 + abs2(vec[3]) * bc5
end

function lambda_epsilon(k, rs)
    return norm(k)^2 / rs^2
end

function num_mBZ(shells)
    return 1 + 3 * shells * (shells + 1)
end

function lambda_ff(V, lambda, k1, k2)
    psi1 = lambda_spinor(lambda, k1)
    psi2 = lambda_spinor(lambda, k2)
    return dot(psi1, V * psi2)
end

# calculates all possible combinations of g1 and g2 that land within the mBZ shells
function sgn_parts(n)
    pairs = []
    for a in -n:n
        for b in -n:n
            if -n <= a + b <= n
                push!(pairs, (a, b))
            end
        end
    end
    return pairs
end


function part_dict(shells)
    dict = Dict{NTuple{2,Int}, Int}()
    parts = sgn_parts(shells)
    for j in 1:length(parts)
        dict[parts[j]] = j
    end
    return dict
end

function lambda_ham!(g1, g2, shell_parts, partitions, part_1s, dict, ham, k, lambda, rs, V)
    # connect all lower order mBZ's to their nn
    for i in 1:length(partitions)
        part_i = partitions[i]
        idex = dict[part_i]
        for j in 1:7
            part_j = part_1s[j]
            gi = part_i[1] * g1 + part_i[2] * g2
            gj = gi .+ part_j[1] * g1 + part_j[2] * g2
            jdex = dict[part_i .+ part_j]
            if part_j != (0, 0)
                val = 1/2 * lambda_ff(V, lambda, k .+ gi, k .+ gj)
                ham[idex, jdex] = val
                ham[jdex, idex] = conj(val)
            end
        end
    end
    # connect outside ring of mBZ's together
    outer_ring = setdiff(shell_parts, partitions)
    for i in 1:length(outer_ring)
        part_i = outer_ring[i]
        idex = dict[part_i]
        for j in 1:7
            part_j = part_1s[j]
            gi = part_i[1] * g1 + part_i[2] * g2
            gj = gi .+ part_j[1] * g1 + part_j[2] * g2
            if (part_i .+ part_j) in outer_ring && part_j != (0, 0)
                jdex = dict[part_i .+ part_j]
                val = 1/2 * lambda_ff(V, lambda, k .+ gi, k .+ gj)
                ham[idex, jdex] = val
                ham[jdex, idex] = conj(val)
            end
        end
    end
    # add in kinetic terms
    for i in 1:length(shell_parts)
        part = shell_parts[i]
        idex = dict[part]
        gi = part[1] * g1 + part[2] * g2
        ham[idex, idex] = lambda_epsilon(gi .+ k, rs)
    end
    return ham
end


function lambda_ham(k, lambda, rs, kappa, V, shells)
    mBZ_count = num_mBZ(shells)
    ham = zeros(ComplexF64, mBZ_count, mBZ_count)
    shell_parts = sgn_parts(shells)
    partitions = sgn_parts(shells - 1)
    part_1s = sgn_parts(1)
    dict = part_dict(shells)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    lambda_ham!(g1, g2, shell_parts, partitions, part_1s, dict, ham, k, lambda, rs, V)
end

function partitions_to_momenta(partitions, kappa)
    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]
    momenta_list = Array{Float64}(undef, length(partitions), 2)
    for j in 1:length(partitions)
        part = partitions[j]
        momenta_list[j, :] = part[1] * g1 + part[2] * g2
    end
    return momenta_list
end

function lambda_mBZ_bc(k, lambda, rs, kappa, V, shells, index, spacing)
    num_vertices = 4

    g1 = kappa * sqrt(3) * [-sqrt(3)/2, 1/2]
    g2 = kappa * sqrt(3) * [0, 1]

    mBZ_count = num_mBZ(shells)
    shell_parts = sgn_parts(shells)
    partitions = sgn_parts(shells - 1)
    part_1s = sgn_parts(1)
    dict = part_dict(shells)

    ham = zeros(ComplexF64, mBZ_count, mBZ_count)
    
    spinors = Array{ComplexF64}(undef, num_vertices, mBZ_count, 2)
    grounds = Array{ComplexF64}(undef, num_vertices, mBZ_count)

    plaq_area = area(spacing * sqrt(2), num_vertices)
    lambda_mBZ_bc!(g1, g2, num_vertices, plaq_area, shell_parts, partitions, part_1s, dict, ham, spinors, grounds, k, lambda, rs, kappa, V, shells, index, spacing)
end

function lambda_mBZ_bc!(g1, g2, num_vertices, plaq_area, shell_parts, partitions, part_1s, dict, ham, spinors, grounds, k, lambda, rs, kappa, V, shells, index, spacing)
    for j in 1:num_vertices
        x_new = k[1] + spacing * cos(2 * pi * (j - 1) / num_vertices)
        y_new = k[2] + spacing * sin(2 * pi * (j - 1) / num_vertices)
        ham = lambda_ham!(g1, g2, shell_parts, partitions, part_1s, dict, ham, [x_new, y_new], lambda, rs, V)
        grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, index]))
        for m in 1:length(shell_parts)
            momentum = g1 * shell_parts[m][1] + g2 * shell_parts[m][2] + [x_new, y_new]
            spinors[j, m, :] = lambda_spinor(lambda, momentum)
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
