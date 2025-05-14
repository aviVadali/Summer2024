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
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for j in 0:(Nl - 1)
        sum1 += nu^(2*j) * kappa^(2*j)
        sum2 += j * nu^(2*j) * kappa^(2*j - 1)
        sum3 += omega^j * nu^(2*j) * kappa^(2*j)
        sum4 += 2 * j * omega * omega^j * nu^(2*j) * kappa^(2*j - 1) 
    end
    return (sum2 * sum3)/sum1^2 + sum4/sum1
end
function closed_rmg_alpha_all(kappa, nu, Nl)
    omega = exp(im * 2 * pi / 3)
    return (1/kappa * ((1 - nu^2 * kappa^2)/(1 - (nu^2 * kappa^2)^Nl))^2 * (1/(1 - nu^2 * kappa^2)^2 * 
    (nu^2 * kappa^2 - Nl * (nu^2 * kappa^2)^Nl + (Nl - 1) * (nu^2 * kappa^2)^(Nl + 1))) * 
    (1/(1 - omega * nu^2 * kappa^2) * (1 - (omega * nu^2 * kappa^2)^Nl)) + 2*omega/kappa * 
    (1 - nu^2 * kappa^2)/(1 - (nu^2 * kappa^2)^Nl) * 1/(1 - omega * nu^2 * kappa^2)^2 * 
    (omega * nu^2 * kappa^2 - Nl * (omega * nu^2 * kappa^2)^Nl + (Nl - 1) * (omega * nu^2 * kappa^2)^(Nl + 1)))
end
function limit_alpha_all(kappa, Nl)
    omega = exp(im * 2 * pi / 3)
    return (Nl - 1)/kappa * im * sqrt(3) * omega^(Nl - 1)
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
        sum1 += nu^(2*l) * kappa^(2*l) * omega^l * l
        sum2 += nu^(2*l) * kappa^(2*l) * omega^l
    end
    return (-1/kappa) * (2 * conj(omega) * sum1 + sum2 * sum3 / sum4) / denom
end
function limit_alpha_l(kappa, nu, Nl, l)
    m = l[1]
    omega = exp(im * 2 * pi / 3)
    return -1/kappa * (nu * kappa)^(2 * (m - Nl)) * (2 * (m - 1) * omega^(m - 2) + (Nl - 1) * omega^(m - 1))
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

function rmg_delta_linear(kappa, nu, Nl)
    omega = exp(im * 2 * pi / 3)
    dn = 10^(-16)
    nm = 10^(-16)
    for j in 0:(Nl - 1)
        dn += nu^(2*j) * kappa^(2*j)
        nm += nu^(2*j) * kappa^(2*j) * omega^j * (j)
    end
    return nm / dn
end

function rmg_alpha_linear(kappa, nu, Nl)
    omega = exp(im * 2 * pi / 3)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for j in 0:(Nl - 1)
        sum1 += nu^(2*j) * kappa^(2*j)
        sum2 += j * nu^(2*j) * kappa^(2*j - 1)
        sum3 += omega^j * nu^(2*j) * kappa^(2*j) * (j)
        sum4 += 2 * j * omega * omega^j * nu^(2*j) * kappa^(2*j - 1) * (j)
    end
    return (sum2 * sum3)/sum1^2 + sum4/sum1
end

function rmg_delta_exp(kappa, nu, Nl)
    omega = exp(im * 2 * pi / 3)
    dn = 10^(-16)
    nm = 10^(-16)
    for j in 0:(Nl - 1)
        dn += nu^(2*j) * kappa^(2*j)
        nm += nu^(2*j) * kappa^(2*j) * omega^j * exp(3*j)
    end
    return nm / dn
end

function rmg_alpha_exp(kappa, nu, Nl)
    omega = exp(im * 2 * pi / 3)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for j in 0:(Nl - 1)
        sum1 += nu^(2*j) * kappa^(2*j)
        sum2 += j * nu^(2*j) * kappa^(2*j - 1)
        sum3 += omega^j * nu^(2*j) * kappa^(2*j) * exp(3*j)
        sum4 += 2 * j * omega * omega^j * nu^(2*j) * kappa^(2*j - 1) * exp(3*j)
    end
    return (sum2 * sum3)/sum1^2 + sum4/sum1
end

# function rmg_delta_linear_layer(kappa, nu, Nl, layer)
#     omega = exp(im * 2 * pi / 3)
#     nmz = 10^(-16)
#     num = 10^(-16)
#     for j in 0:(Nl - 1)
#         nmz += nu^(2*j) * kappa^(2*j)
#     end
#     l = layer - 1
#     num = -(l + 1) * nu^(2*l) * kappa^(2*l) * omega^l
#     return num / nmz
# end

# function rmg_alpha_linear_layer(kappa, nu, Nl, layer)
#     omega = exp(im * 2 * pi / 3)
#     denom = 10^(-16)
#     sum1 = 10^(-16)
#     sum2 = 10^(-16)
#     sum3 = 10^(-16)
#     sum4 = 10^(-16)
#     for j in 0:(Nl - 1)
#         denom += nu^(2*j) * kappa^(2*j)
#         sum3 += nu^(2*j) * kappa^(2*j) * j
#         sum4 += nu^(2*j) * kappa^(2*j)
#     end
#     l = layer - 1
#     sum1 = nu^(2*l) * kappa^(2*l) * omega^l * l
#     sum2 = nu^(2*l) * kappa^(2*l) * omega^l
#     return -(l + 1) * (-1/kappa) * (2 * conj(omega) * sum1 + sum2 * sum3 / sum4) / denom
# end

# function rmg_delta_exp_layer(kappa, nu, Nl, layer)
#     omega = exp(im * 2 * pi / 3)
#     nmz = 10^(-16)
#     num = 10^(-16)
#     for j in 0:(Nl - 1)
#         nmz += nu^(2*j) * kappa^(2*j)
#     end
#     l = layer - 1
#     num = -exp(-l - 1) * nu^(2*l) * kappa^(2*l) * omega^l
#     return num / nmz
# end

# function rmg_alpha_exp_layer(kappa, nu, Nl, layer)
#     omega = exp(im * 2 * pi / 3)
#     denom = 10^(-16)
#     sum1 = 10^(-16)
#     sum2 = 10^(-16)
#     sum3 = 10^(-16)
#     sum4 = 10^(-16)
#     for j in 0:(Nl - 1)
#         denom += nu^(2*j) * kappa^(2*j)
#         sum3 += nu^(2*j) * kappa^(2*j) * j
#         sum4 += nu^(2*j) * kappa^(2*j)
#     end
#     l = layer - 1
#     sum1 = nu^(2*l) * kappa^(2*l) * omega^l * l
#     sum2 = nu^(2*l) * kappa^(2*l) * omega^l
#     return -exp(-l - 1) * (-1/kappa) * (2 * conj(omega) * sum1 + sum2 * sum3 / sum4) / denom
# end


# function rmg_inf_layer_delta(kappa, nu, layer, g)
#     omega = exp(im * 2 * pi / 3)
#     l = layer - 1
#     return omega^l * g * (kappa * nu)^(2 * l) * (1 - kappa^2 * nu^2)
# end

# function rmg_inf_layer_alpha(kappa, nu, layer, g)
#     omega = exp(im * 2 * pi / 3)
#     l = layer - 1
#     return ((-1/kappa) * omega^l * g * (kappa * nu)^(2 * l) * (kappa^2 * nu^2 + 
#     2 * conj(omega) * l * (1 - kappa^2 * nu^2))) 
# end

