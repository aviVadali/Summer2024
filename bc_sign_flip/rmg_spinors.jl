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
function rmg_patch_bc(points, spacing, a, uD, t0, t1, t2, t3, t4, Nl, index, kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
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
                k_pt = [kappa * cos(2*pi/3 * (m - 1)), kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                sor = normalize(rmg_spinor(a, uD, t0, t1, t2, t3, t4, Nl, k_pt, index))
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