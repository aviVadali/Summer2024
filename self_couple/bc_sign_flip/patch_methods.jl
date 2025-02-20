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
    return mat0 + mat1
end

# we can choose nv arbitrarily
# just need to fix a smooth gauge
# function gauge_fix(state)
#     dim = length(state)
#     nv = ones(dim)
#     phase = angle(dot(nv, state))
#     return state * exp(-phase * im)
# end

function gauge_fix(state)
    entry = state[1]
    phi = angle(entry / abs(entry))
    state = state .* exp(-im * phi)
    return state
end


function spinor_inner(C1, C2, q1_spinor, q2_spinor)
    val = 0
    for m in 1:3
        val += conj(C1[m]) * C2[m] * dot(q1_spinor[m, :], q2_spinor[m, :])
    end
    return val
end