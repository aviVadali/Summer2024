function f(k, w)
    if k < 1 - w/2
        return 0
    elseif 1 - w/2 <= k <= 1 + w/2
        return k/w - 1/w + 1/2
    elseif 1 + w/2 < k
        return 1
    end
end
function spinor(k, theta, n, w)
    return [1, f(k, w) * exp(im * n * theta)]
end
# Berry curvature over all plaquettes
function sp_berry_curvature(points, spacing, n, w)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through square plaquette centered at point
        angles = Array{Float64}(undef, 4)
        momenta = Array{Float64}(undef, length(angles))
        for j in 1:length(angles)
            x_new = points[i, 1] + spacing * cos(2 * pi * (j - 1) / length(angles))
            y_new = points[i, 2] + spacing * sin(2 * pi * (j - 1) / length(angles))
            momenta[j] = norm([x_new, y_new])
            angles[j] = polar_angle(x_new, y_new)
        end

        states = Array{ComplexF64}(undef, length(angles), 2)
        for j in 1:length(angles)
            states[j, :] = normalize(spinor(momenta[j], angles[j], n, w))
        end
        P = 1
        for j in 1:length(angles)
            if j < length(angles)
                P *= dot(states[j, :], states[j + 1, :]) / abs(dot(states[j, :], states[j + 1, :]))
            else
                P *= dot(states[j, :], states[1, :]) / abs(dot(states[j, :], states[1, :]))
            end
        end
        berry_list[i] = -angle(P) / area(spacing, length(angles))
    end
    return berry_list
end
function H_k(q, theta, vF)
    n1 = [cos(0), sin(0)]
    n3 = [cos(2 * pi / 3), sin(2 * pi / 3)]
    n5 = [cos(4 * pi / 3), sin(4 * pi / 3)]
    k = vF * q * [cos(theta), sin(theta)]
    return [dot(k, n1) 0 0; 0 dot(k, n3) 0; 0 0 dot(k, n5)]
end
function H_mft(q, theta, delt, alph)
    # useful phase
    omega = exp(im * 2 * pi / 3)
    mat0 = [0 delt conj(delt); 
            conj(delt) 0 delt; 
            delt conj(delt) 0]
    qq = q * cos(theta) + im * q * sin(theta)
    mat1 = [0 alph * (omega * qq + conj(omega) * conj(qq)) conj(alph) * (omega^2 * qq + conj(omega^2) * conj(qq));
             conj(alph) * (omega * qq + conj(omega) * conj(qq)) 0 alph * (qq + conj(qq));
             alph * (omega^2 * qq + conj(omega^2) * conj(qq)) conj(alph) * (qq + conj(qq)) 0]
    return mat0 + mat1
end
# Berry curvature over all plaquettes
function mf_berry_curvature(points, spacing, n, w, vF, kappa, delt, alph, pure_pos = false)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        angles = Array{Float64}(undef, 4)
        momenta = Array{Float64}(undef, length(angles))
        states = Array{ComplexF64}(undef, length(angles), 3, 2)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:length(angles)
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / length(angles))
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / length(angles))
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            if pure_pos == true
                gs = [1 1 1]
            else
                gs = eigvecs(Hermitian(ham))[:, 1]
            end
            for m in 1:3
                k_pt = [kappa * cos(2*pi/3 * (m - 1)), kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                mom = norm(k_pt)
                ang = polar_angle(k_pt[1], k_pt[2])
                states[j, m, :] = gs[m] * normalize(spinor(mom, ang, n, w))
            end
        end
        P = 1
        for j in 1:length(angles)
            if j < length(angles)
                temp = dot(states[j, 1, :], states[j + 1, 1, :]) + dot(states[j, 2, :], states[j + 1, 2, :]) + 
                dot(states[j, 3, :], states[j + 1, 3, :])
                P *= temp
            else
                temp = dot(states[j, 1, :], states[1, 1, :]) + dot(states[j, 2, :], states[1, 2, :]) + 
                dot(states[j, 3, :], states[1, 3, :])
                P *= temp
            end
            if temp != 0
                P /= abs(temp)
            end
        end
        berry_list[i] = -angle(P) / area(spacing, length(angles))
    end
    
    return berry_list
end
function BC_PT(alpha, delta, n, w, p, theta)
    if theta == 0
        val = (1/(2025 * w * delta^4)) * (
            36 * n * p^4 * (alpha^4 - 
            14 * abs(alpha)^4 + conj(alpha)^4) + 
            10 * p^2 * w * (-27 * n * alpha * delta * abs(alpha)^2 - 
            27 * n * alpha * delta * conj(alpha)^2 + (-40im * sqrt(3) * alpha + 9 * n * delta) * conj(alpha)^3 - 
            20im * sqrt(3) * conj(alpha)^4 + alpha^3 * (60im * sqrt(3) * alpha + 
            9 * n * delta + 80 * sqrt(3) * imag(alpha))) + 
            18 * delta^2 * (-72 * n * delta^2 + 
            25im * sqrt(3) * w * (alpha^2 - conj(alpha)^2) + 
            30 * n * w * delta * real(alpha)) + 
            5 * n * p^3 * (-6 * w * abs(alpha)^4 - 
            2 * (5 * w * alpha + 48 * delta) * conj(alpha)^3 - 
            7 * w * conj(alpha)^4 + alpha^3 * (3 * w * alpha - 
            96 * delta - 20 * w * real(alpha)))
        )
    elseif theta == pi/4
        val = (1/(8100 * w^2 * delta^4)) * (
            144 * n * p^4 * w * (alpha^4 - 
            14 * abs(alpha)^4 + conj(alpha)^4) + 
            40 * p^2 * w^2 * (-27 * n * alpha * delta * abs(alpha)^2 - 
            27 * n * alpha * delta * conj(alpha)^2 + (-40im * sqrt(3) * alpha + 9 * n * delta) * conj(alpha)^3 - 
            20im * sqrt(3) * conj(alpha)^4 + alpha^3 * (60im * sqrt(3) * alpha + 
            9 * n * delta + 80 * sqrt(3) * imag(alpha))) + 
            72 * w * delta^2 * (-72 * n * delta^2 + 
            25im * sqrt(3) * w * (alpha^2 - conj(alpha)^2) + 
            30 * n * w * delta * real(alpha)) + 
            sqrt(2) * p^3 * (5 * alpha^3 * (14 * n * w^2 * alpha + 
            3 * (48im + n * w * (64 + 9im * n * w)) * delta) + 
            27im * (-16 + 
            5 * n^2 * w^2) * alpha * delta * abs(alpha)^2 + 
            60 * n * w^2 * abs(alpha)^4 + 
            2 * conj(alpha) * (50 * n * w^2 * alpha^3 + 
            conj(alpha) * (-576im * alpha * delta + 
            5 * n * w * conj(alpha) * (10 * w * alpha + 
            96 * delta + 7 * w * conj(alpha)) + 
            45im * (16 + 3 * n^2 * w^2) * delta * real(alpha)))
        ))
        
    end
    return real(val)
end
# Berry curvature over all plaquettes
function bc_no_spinors(points, spacing, vF, kappa, delt, alph, pure_pos = false)
    berry_list = Array{Float64}(undef, size(points, 1))
    P_list = Array{ComplexF64}(undef, size(points, 1))
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        angles = Array{Float64}(undef, 4)
        momenta = Array{Float64}(undef, length(angles))
        states = Array{ComplexF64}(undef, length(angles), 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:length(angles)
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / length(angles))
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / length(angles))
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            if pure_pos == true
                gs = [1 1 1]
            else
                gs = eigvecs(Hermitian(ham))[:, 1]
            end
            states[j, :] = normalize(gs)
        end
        P = 1
        for j in 1:length(angles)
            if j < length(angles)
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
        P_list[i] = P
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing, length(angles))
        else
            berry_list[i] = -angle(P) / area(spacing, length(angles))
        end
    end
    return berry_list, P_list
end
function state_coefficients(p, theta, kappa, alpha, delta, index)
    mat = H_mft(p, theta, delta, alpha)
    gs = eigvecs(Hermitian(mat))[:, index]
    return gs, eigvals((Hermitian(mat)))
end