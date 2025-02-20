function mesh_grid(list1, list2, type)
    l1 = length(list1)
    l2 = length(list2)
    grid = Array{type}(undef, l1 * l2, 2)
    for j in 1:l1
        grid[1 + (j-1)*l2:j*l2, 1] = ones(l2) * list1[j]
        grid[1 + (j-1)*l2:j*l2, 2] = list2
    end
    return grid
end
function rmg_delta_all(kappa, nu, Nl)
    omega = exp(im * 2 * pi / 3)
    dn = 10^(-16)
    nm = 10^(-16)
    for j in 0:(Nl - 1)
        dn += nu^(2*j) * kappa^(2*j)
        nm += nu^(2*j) * kappa^(2*j) * omega^j
    end
    return nm / dn
end
function limit_delta_all(Nl)
    omega = exp(im * 2 * pi / 3)
    return omega^(Nl - 1)
end
function rmg_alpha_all(kappa, nu, Nl)
    omega = exp(im * 2 * pi / 3)
    denom = 10^(-16)
    sum1 = 10^(-16)
    sum2 = 10^(-16)
    sum3 = 10^(-16)
    sum4 = 10^(-16)
    for j in 0:(Nl - 1)
        denom += nu^(2*j) * kappa^(2*j)
        sum1 += nu^(2*j) * kappa^(2*j) * omega^j
        sum2 += nu^(2*j) * kappa^(2*j) * omega^j
        sum3 += nu^(2*j) * kappa^(2*j) * j
        sum4 += nu^(2*j) * kappa^(2*j)
    end
    return (-1/kappa) * (2 * conj(omega) * sum1 + sum2 * sum3 / sum4) / denom
end
function limit_alpha_all(kappa, Nl)
    omega = exp(im * 2 * pi / 3)
    return -1/kappa * (2 * omega^(Nl - 2) + (Nl - 1) * omega^(Nl - 1))
end
function rmg_delta_layers(kappa, nu, Nl, layers)
    omega = exp(im * 2 * pi / 3)
    nmz = 10^(-16)
    num = 10^(-16)
    for j in 0:(Nl - 1)
        nmz += nu^(2*j) * kappa^(2*j)
    end
    for j in eachindex(layers)
        l = layers[j] - 1
        num += nu^(2*l) * kappa^(2*l) * omega^l
    end
    return num / nmz
end
function limit_delta_l(kappa, nu, Nl, l)
    m = l[1]
    omega = exp(im * 2 * pi / 3)
    return (nu * kappa)^(2 * (m - Nl)) * omega^(m - 1)
end
function rmg_alpha_layers(kappa, nu, Nl, layers)
    omega = exp(im * 2 * pi / 3)
    denom = 10^(-16)
    sum1 = 10^(-16)
    sum2 = 10^(-16)
    sum3 = 10^(-16)
    sum4 = 10^(-16)
    for j in 0:(Nl - 1)
        denom += nu^(2*j) * kappa^(2*j)
        sum3 += nu^(2*j) * kappa^(2*j) * j
        sum4 += nu^(2*j) * kappa^(2*j)
    end
    for j in eachindex(layers)
        l = layers[j] - 1
        sum1 += nu^(2*l) * kappa^(2*l) * omega^l
        sum2 += nu^(2*l) * kappa^(2*l) * omega^l
    end
    return (-1/kappa) * (2 * conj(omega) * sum1 + sum2 * sum3 / sum4) / denom
end
function limit_alpha_l(kappa, nu, Nl, l)
    m = l[1]
    omega = exp(im * 2 * pi / 3)
    return -1/kappa * (nu * kappa)^(2 * (m - Nl)) * (2 * omega^(m - 2) + (Nl - 1) * omega^(m - 1))
end

function rmg_gauge_transform(alpha, delta, kappa, C)
    return alpha + 1im * delta * (-3 * sqrt(3) * kappa) * C
end
function rmg_C_phi(alpha, delta, kappa, phi)
    return (cos(phi) * imag(alpha) - sin(phi) * real(alpha)) / 
    (3 * sqrt(3) * kappa^2 * (cos(phi) * real(delta) + sin(phi) * imag(delta)))
end
function bc_sign_alpha_delta(m_kappa, spacing, Nl, nu, alpha, delta, q)
    grid = reshape([q[1], q[2]], (1, 2))
    patch_bc = rmg_patch_bc(grid, spacing, Nl, nu, m_kappa, vF, delta, alpha)[1]
    spinor_bc = rmg_spinor_bc(grid, spacing, Nl, nu)[1]
    diff = abs(patch_bc + spinor_bc)
    if diff < patch_bc
        return true
    else
        return false
    end
end