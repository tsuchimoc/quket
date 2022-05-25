# Copyright 2022 The Quket Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# limitations under the License.
import time

import scipy as sp
import numpy as np
from numpy import linalg as LA
from qulacs import QuantumState
from qulacs.state import inner_product
from openfermion.transforms import reverse_jordan_wigner
try:
    from openfermion.utils import normal_ordered
except:
    from openfermion.transforms import normal_ordered

from .qite_function import (conv_anti, anti_to_base, make_gate,
                            calc_delta, calc_psi_lessH, qite_s_operators,
                            hamiltonian_to_str, calc_inner2, qlanczos,multiply_Hpauli,calc_expHpsi)
from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints, print_state, printmat
from quket.utils import fermi_to_str
from quket.linalg import lstsq, root_inv


def msite(Quket, id_set, size):
    # Parameter setting
    nstates = len(Quket.multi.weights)
    n = Quket.n_qubits
    for istate in range(nstates):
        state=Quket.multi.states[istate]
        prints(f"Multi state[{istate}]: |{format(state,'b').zfill(n)} >")
        prints(f"Multi weights[{istate}]: {Quket.multi.weights[istate]} ")
    ansatz = Quket.ansatz
    if ansatz=="cite":
        prints(f"MSITE")
    else:
        prints(f"MSQITE")
    db = Quket.dt
    qbit = Quket.current_det
    ntime = Quket.maxiter
    observable = Quket.qulacs.Hamiltonian
    #observable2 = Quket.qulacs.Hamiltonian2
    S2_observable = Quket.qulacs.S2
    Number_observable = Quket.qulacs.Number
    threshold = Quket.ftol
    S2 = 0
    Number = 0
    use_qlanczos = Quket.qlanczos

    prints(f"QITE: Pauli operator group size = {size}")
    if ansatz != "cite":
        sigma_list, sigma_ij_index, sigma_ij_coef = qite_s_operators(id_set, n)
        len_list = len(sigma_list)
        prints(f"    Unique sigma list = {len_list}")
    index = np.arange(n)
    

    first_state = [QuantumState(n) for x in range(nstates)]
    for istate in range(nstates):
        first_state[istate].set_computational_basis(Quket.multi.states[istate])

    delta = [QuantumState(n) for x in range(nstates)]
    energy = [[] for x in range(nstates)]
    psi_dash = [first_state[x].copy() for x in range(nstates)]
    En = [0 for x in range(nstates)]
    S2 = [0 for x in range(nstates)]
    Number = [0 for x in range(nstates)]
    cm = [0 for x in range(nstates)]

    # pramater of lanczos #
    s2 = []
    nm_list = []
    cm_list = []
    nm_list.append(1)
    cm_list.append(1)
    qlanz = []
    ##########
    
    t1 = time.time()
    cf.t_old = t1
    for istate in range(nstates):
        En[istate] = observable.get_expectation_value(psi_dash[istate])
        energy[istate].append(En[istate])

    # pramater of lanczos #
    q_en = [En]
    H_q = None
    S_q = None
    S2_q = None
    for istate in range(nstates):
        if S2_observable is not None:
            S2[istate] = S2_observable.get_expectation_value(psi_dash[istate])
        else:
            S2 = 0
        if Number_observable is not None:
            Number[istate] = Number_observable.get_expectation_value(psi_dash[istate])
    ##########

    ##########
    # Shift  #
    ##########
    if Quket.shift in ['hf', 'step']:
        shift = [En[x] for x in range(nstates)]
    elif Quket.shift == 'true':
        shift = [En[x] for x in range(nstates)]
    elif Quket.shift in ['none', 'false']:
        shift = [0 for x in range(nstates)]
    else:
        raise ValueError(f"unknown shift option: {Quket.shift}")
    order = 1
    prints('Shift = ', shift)
    prints('QLanczos = ', use_qlanczos)
    ref_shift = shift  # Reference shift
    dE = 100
    beta = 0
    Conv = False
    c = np.zeros((nstates,nstates))
    for t in range(ntime):
        t2 = time.time()
        cput = t2 - cf.t_old
        cf.t_old = t2
        if cf.debug:
            for istate in range(nstates):
                print_state(psi_dash[istate], name=f'istate={istate}')
        for istate in range(nstates):
            if istate == 0:
                prints(f"{beta:6.2f}")
            prints(f"state{istate}: E = {En[istate]:.12f}  "
                f"<S**2> = {S2[istate]:+10.8f}  "
                f"<N> = {Number[istate]:10.8f}  "
                f"Fidelity = {Quket.fidelity(psi_dash[istate]):.6f}  "
                f"CPU Time = {cput: 5.2f}  ")
        """
        for istate in range(nstates):
            if mpi.main_rank:
                prints(f"state{istate}")
                print_state(psi_dash[istate])
        """    
        prints("")
        # Save the QLanczos details in out

        if abs(dE) < threshold:
            Conv = True
            break
        if t == 0:
            xv = [np.zeros(size) for x in range(nstates)]
        if Quket.shift == 'step':
            shift = [En[x] for x in range(nstates)]
        elif Quket.shift == 'none':
            shift = [0 for x in range(nstates)]
        T0 = time.time()
        
        beta += db
        T1 = time.time()
        ####
        #MSITE
        #ξIとξJからSeffを求める　ξ=psi_dash
        S_eff= np.zeros((nstates,nstates), dtype=complex)
        for i in range(nstates):
            for j in range(nstates):
                #<ξI(l)|H|ξJ(l)>
                seff_ij=observable.get_transition_amplitude(psi_dash[i],psi_dash[j])
                #test=observable2.get_transition_amplitude(psi_dash[i],psi_dash[j])
                #seff_j, notuse = calc_delta(psi_dash[j], observable, n,
                #                                2*db, shift=0, order=order, ref_shift=0)
                #seff_ij=inner_product(psi_dash[i], seff_j)
                if i==j:
                    #S_eff[i][j]=1-2*db*seff_ij
                    #prints(f'{db=},  <i|H|j>={seff_ij},  <i|H**2|j>={test}')
                    S_eff[i,j]=1-2*db*(seff_ij - shift[i])
                else:
                    S_eff[i,j]=-2*db*seff_ij
        
        d = Lowdin_orthonormalization(S_eff)
#        #Seffを対角化
#        seff_eig,u = np.linalg.eig(S_eff)
#        
#        #s^(-1/2)
#        seff_eig_2=np.diag(seff_eig)
#        seff_eig_2 = np.linalg.pinv(seff_eig_2)
#        seff_eig_2 = sp.linalg.sqrtm(seff_eig_2)
#       
#        #d=us^(-1/2)u†
#        d=u@seff_eig_2@np.conjugate(u.T)
#        printmat(d, name='d')
#
        #ξ(l+1)をdを使って求める
        psi_next=[QuantumState(n) for x in range(nstates)]
        for i in range(nstates):
            state_i=QuantumState(n)
            state_i.multiply_coef(0)
            #for j in range(nstates):
            #    state_j=psi_dash[j].copy()
            #    state_j.multiply_coef(d[i][j])
            #    state_i.add_state(state_j)
            #next_i, norm = calc_expHpsi(state_i, observable, n,
            #                            db, shift=shift[i], order=order, ref_shift=ref_shift[i])
            ##prints(f'{norm=}', ' ??? ', shift[i] , 1 - 2*seff_ij*db + 2*shift[i]*db,  1-2*db*(seff_ij-shift[i]) + db*db*(test - 2*seff_ij*shift[i] + shift[i]**2))
            #prints(f'{norm=}')
            #psi_next[i]=next_i
            for j in range(nstates):
                state_j=psi_dash[j].copy()
                state_j, norm = calc_expHpsi(state_j, observable, n,
                                        db, observable.get_expectation_value(state_j), order=order, ref_shift=ref_shift[j])
                state_j.multiply_coef(d[i][j])
                state_i.add_state(state_j)

#            state_i, norm = calc_expHpsi(state_i, observable, n,
#                                        db, observable.get_expectation_value(state_i), order=order, ref_shift=ref_shift[j])
            #prints(f'{norm=}', ' ??? ', shift[i] , 1 - 2*seff_ij*db + 2*shift[i]*db,  1-2*db*(seff_ij-shift[i]) + db*db*(test - 2*seff_ij*shift[i] + shift[i]**2))
            psi_next[i]=state_i.copy()
#        for i in range(nstates):
#            for j in range(nstates):
#                S_eff[j,i] = inner_product(psi_next[i],psi_next[j])
#
#        #printmat(S_eff, name='Seff??')
#        d = Lowdin_orthonormalization(S_eff)
#        #printmat(d@S_eff@d)
#        
#        psi_nextX=[QuantumState(n) for x in range(nstates)]
#        for i in range(nstates):
#            state_i=QuantumState(n)
#            state_i.multiply_coef(0)
#            for j in range(nstates):
#                state_j = psi_next[j].copy()
#                state_j.multiply_coef(d[i,j])
#                state_i.add_state(state_j)
#            psi_nextX[i] = state_i.copy()
#        for i in range(nstates):
#            psi_next[i] = psi_nextX[i].copy()
#        for i in range(nstates):
#            for j in range(nstates):
#                S_eff[j,i] = inner_product(psi_nextX[i],psi_nextX[j])
#        #printmat(S_eff)
#                
#            
#

        if ansatz!= "cite":
            #Aを選ぶ (S+ST)a+b=0
            #Δ0
            delta=[QuantumState(n) for x in range(nstates)]
            for istate in range(nstates):
                delta[istate]=psi_next[istate].copy()
                p=psi_dash[istate].copy()
                p.multiply_coef(-1)
                delta[istate].add_state(p)
                
            # Compute Sij as expectation values of sigma_list
            Sij_list = [np.zeros(len_list) for x in range(nstates)]
            Sij_my_list = [np.zeros(len_list) for x in range(nstates)]
            ipos, my_ndim = mpi.myrange(len_list)
            T2 = time.time()
            for iope in range(ipos, ipos+my_ndim):
                for istate in range(nstates):
                    val = sigma_list[iope].get_expectation_value(psi_dash[istate])
                    Sij_my_list[istate][iope] = val
            T3 = time.time()
            for istate in range(nstates):
                mpi.comm.Allreduce(Sij_my_list[istate], Sij_list[istate], mpi.MPI.SUM)
            T4 = time.time()

            # Distribute Sij
            ij = 0
            sizeT = size*(size-1)//2
            ipos, my_ndim = mpi.myrange(sizeT)
            S = [np.zeros((size, size), dtype=complex)  for x in range(nstates)]
            my_S = [np.zeros((size, size), dtype=complex)  for x in range(nstates)]
            for i in range(size):
                for j in range(i):
                    if ij in range(ipos, ipos+my_ndim):
                        for istate in range(nstates):
                            idx = sigma_ij_index[ij]
                            coef = sigma_ij_coef[ij]
                            my_S[istate][i, j] = coef*Sij_list[istate][idx]
                            my_S[istate][j, i] = my_S[istate][i, j].conjugate()

                    ij += 1
            for istate in range(nstates):
                mpi.comm.Allreduce(my_S[istate], S[istate], mpi.MPI.SUM)
            for i in range(size):
                for istate in range(nstates):
                    S[istate][i, i] = 1

            T5 = time.time()
            sigma = [[] for x in range(nstates)]
            for i in range(size):
                for istate in range(nstates):
                    pauli_id = id_set[i]
                    circuit_i = make_gate(n, index, pauli_id)
                    state_i = psi_dash[istate].copy()
                    circuit_i.update_quantum_state(state_i)
                    sigma[istate].append(state_i)
            T6 = time.time()

            b_l = [np.zeros(size) for x in range(nstates)]
            for i in range(size):
                for istate in range(nstates):
                    b_i = inner_product(sigma[istate][i], delta[istate])
                    b_l[istate][i] = -2*b_i.imag
            Amat = [2*np.real(S[x]) for x in range(nstates)]
            T7 = time.time()

            b_sum = np.zeros(size)
            A_mat_sum = np.zeros((size, size))
            for istate in range(nstates):
                b_sum += b_l[istate]
                A_mat_sum += Amat[istate]

        ### Sum ###
            independent = False
            if independent == False:
                a, res, rnk, s = lstsq(A_mat_sum, b_sum, cond=1e-6)
                for istate in range(nstates):
                    psi_next[istate] = calc_psi_lessH(psi_dash[istate], n, index, a, id_set)

            else:
        ### Independent ###
                for istate in range(nstates):
                    a, res, rnk, s = lstsq(Amat[istate], b_l[istate], cond=1e-6)
                    psi_next[istate] = calc_psi_lessH(psi_dash[istate], n, index, a, id_set)
            """
            for istate in range(nstates):
                x, res, rnk, s = lstsq(Amat[istate], b_l[istate], cond=1e-8)
                a = x.copy()
                # Just in case, broadcast a...
                mpi.comm.Bcast(a, root=0)  
                psi_next[istate] = calc_psi_lessH(psi_dash[istate], n, index, a, id_set)
            """
#            #Aを選ぶ (S+ST)a+b=0
#            #Δ0 = \sum_I Δ0(I)
#            delta = QuantumState(n)
#            delta.multiply_coef(0)
#            for istate in range(nstates):
#                delta.add_state(psi_next[istate])
#                p = psi_dash[istate].copy()
#                p.multiply_coef(-1)
#                delta.add_state(p)
#            psi_sum = QuantumState(n)
#            psi_sum.multiply_coef(0)
#            for istate in range(nstates):
#                psi_sum.add_state(psi_dash[istate])
#                
#            # Compute Sij as expectation values of sigma_list
#            Sij_list = np.zeros(len_list)
#            Sij_my_list = np.zeros(len_list) 
#            ipos, my_ndim = mpi.myrange(len_list)
#            T2 = time.time()
#            for iope in range(ipos, ipos+my_ndim):
#                val = sigma_list[iope].get_expectation_value(psi_sum)
#                Sij_my_list[iope] = val
#            T3 = time.time()
#            mpi.comm.Allreduce(Sij_my_list, Sij_list, mpi.MPI.SUM)
#            T4 = time.time()
#
#            # Distribute Sij
#            ij = 0
#            sizeT = size*(size-1)//2
#            ipos, my_ndim = mpi.myrange(sizeT)
#            S = np.zeros((size, size), dtype=complex)
#            my_S = np.zeros((size, size), dtype=complex) 
#            for i in range(size):
#                for j in range(i):
#                    if ij in range(ipos, ipos+my_ndim):
#                        idx = sigma_ij_index[ij]
#                        coef = sigma_ij_coef[ij]
#                        my_S[i, j] = coef*Sij_list[idx]
#                        my_S[j, i] = my_S[i, j].conjugate()
#
#                    ij += 1
#            mpi.comm.Allreduce(my_S, S, mpi.MPI.SUM)
#            for i in range(size):
#                S[i, i] = nstates
#
#            T5 = time.time()
#            sigma = [] 
#            for i in range(size):
#                pauli_id = id_set[i]
#                circuit_i = make_gate(n, index, pauli_id)
#                state_i = psi_sum.copy()
#                circuit_i.update_quantum_state(state_i)
#                sigma.append(state_i)
#            T6 = time.time()
#
#            b_l = np.zeros(size)
#            for i in range(size):
#                b_i = inner_product(sigma[i], delta)
#                b_l[i] = -2*b_i.imag
#            Amat = 2*np.real(S) 
#            #printmat(Amat,name='Amat')
#            #printmat(b_l,name='b_l')
#            T7 = time.time()
#
#            a, res, rnk, s = lstsq(Amat, b_l, cond=1e-6)
#            for istate in range(nstates):
#                psi_next[istate] = calc_psi_lessH(psi_dash[istate], n, index, a, id_set)
#            """
#            for istate in range(nstates):
#                x, res, rnk, s = lstsq(Amat[istate], b_l[istate], cond=1e-8)
#                a = x.copy()
#                # Just in case, broadcast a...
#                mpi.comm.Bcast(a, root=0)  
#                psi_next[istate] = calc_psi_lessH(psi_dash[istate], n, index, a, id_set)
#            """
        elif ansatz== "cite":
            ### Orthonormalize the state basis again.
            S_eff= np.zeros((nstates,nstates), dtype=complex)
            for i in range(nstates):
                for j in range(nstates):
                    S_eff[i,j]=inner_product(psi_next[i],psi_next[j])
            d = Lowdin_orthonormalization(S_eff) 
            for i in range(nstates):
                 psi_dash[i] = QuantumState(n)
                 psi_dash[i].multiply_coef(0)
                 for j in range(nstates): 
                     chi=psi_next[j].copy() 
                     chi.multiply_coef(d[j,i])
                     psi_dash[i].add_state(chi)
            for i in range(nstates):
                psi_next[i] = psi_dash[i].copy()
#        
#        for istate in range(nstates):
#            En[istate] = observable.get_expectation_value(psi_dash[istate])
        

        #Heff
        #H_eff= [np.zeros(nstates, dtype=complex) for x in range(nstates)] 
        H_eff= np.zeros((nstates,nstates), dtype=complex)
        S_eff= np.zeros((nstates,nstates), dtype=complex)
        for i in range(nstates):
            for j in range(nstates):
                H_eff[i,j]=observable.get_transition_amplitude(psi_next[i],psi_next[j])
                S_eff[i,j]=inner_product(psi_next[i],psi_next[j])

        #Heff c = cE (Heffの対角化)
        #eig,c = np.linalg.eig(H_eff)
        #Hc=ScE　(S^(-1)Hの対角化)
        #eig,c = np.linalg.eig(np.linalg.inv(S_eff)@H_eff)

        root_invS = root_inv(S_eff, eps=1e-9)
        H_ortho = root_invS.T@H_eff@root_invS
        eig,c = np.linalg.eig(H_ortho)

        # Sort 
        ind = np.argsort(eig)
        eig = eig[ind]
        c  = c[:, ind]

        ### <S**2> and <N>
        cvec  = root_invS@c
        for istate in range(nstates):
            state = QuantumState(n)
            state.multiply_coef(0)
            for jstate in range(nstates):
                temp = psi_next[jstate].copy()
                temp.multiply_coef(cvec[jstate,istate])
                state.add_state(temp)
            if S2_observable is not None:
                S2[istate] = S2_observable.get_expectation_value(state)
            else:
                S2 = 0
            if Number_observable is not None:
                Number[istate] = Number_observable.get_expectation_value(state)
            
   
#        #cを用いてξのアップデート
#        for i in range(nstates):
#            psi_dash[i]=QuantumState(n)
#            psi_dash[i].multiply_coef(0)
#            for j in range(nstates): 
#                chi=psi_next[j].copy() 
#                chi.multiply_coef(c[j][i])
#                psi_dash[i].add_state(chi)
#        
#        for istate in range(nstates):
#            En[istate] = observable.get_expectation_value(psi_dash[istate])
#            energy[istate].append(En[istate])
        for istate in range(nstates):
            psi_dash[istate] = psi_next[istate].copy()
            En[istate] = eig[istate].real
            energy[istate].append(En[istate])


        ####
        dE = 0
        for istate in range(nstates):
            dE += energy[istate][t+1] - energy[istate][t]
        dE /= nstates
        Quket.state = psi_dash[0]


    if Conv:
        prints(f"CONVERGED at {beta=:.2f}.")
    else:
        prints(f"CONVERGE FAILED.")
    for istate in range(nstates):
        prints(f" Final: E[{ansatz}] = {En[istate]:.12f}  ")

    if use_qlanczos:
        prints(f"  QLanczos:")
        prints('\n'.join(
            f'     E = {e:+.12f} [<S**2> = {s:+.12f}]' for (e, s) in zip(q_en, q_s2)), end="")
        prints("")
    print_state(psi_dash[0], name="(QITE state)")

def Lowdin_orthonormalization(S):
    eig,u = np.linalg.eig(S)
    #s^(-1/2)
    eig_2 = np.diag(eig)
    eig_2 = np.linalg.pinv(eig_2)
    eig_2 = sp.linalg.sqrtm(eig_2)
    #d=us^(-1/2)u†
    return u@eig_2@np.conjugate(u.T)

