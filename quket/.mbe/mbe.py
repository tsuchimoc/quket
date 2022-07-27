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
import os
import re
import time
import math
import pickle
import inspect
from copy import copy, deepcopy
from typing import Dict, Tuple
from itertools import combinations, product
from dataclasses import dataclass, field

import numpy as np
from openfermion.ops import InteractionOperator
from openfermion.linalg import eigenspectrum

from quket import config as cf
from quket.mpilib import mpilib as mpi
from quket.fileio import prints, error, SaveTheta
from quket.lib import QuantumState, s_squared_operator, number_operator

def MBE_driver(vqe):
    """Many-Body Expansion driver.

    Args:
        vqe (Callable): VQE driver.

    Returns:
        wrapper (Callable): MBE driver.

    Author(s): Yuma Shimomoto
    """
    from quket.orbital import oo, orbital_rotation
    from quket.utils import get_ndims, transform_state_jw2bk
    from quket.opelib import create_1body_operator
    from quket.quket_data import QuketData
    from quket.utils import get_func_kwds
    from quket.fileio import set_config


    def wrapper(Quket, *args, **kwds):

        def Send_Receive_Ehandler(Ehandler):
            ############################
            # Send and Receive results #
            ############################
            if mpi.handler_rank:
                Ehandler_list = []
                for i in range(1, mpi.nprocs):
                    tt1 = time.time()
                    Ehandler_list.append(mpi.comm.recv(source=i, tag=0))
                    tt2 = time.time()
                    if tt2 - tt1 > 1:
                        print(f"I ({mpi.rank}) recieved from {i} {time.time()-tt1}.")
                for handler in Ehandler_list:
                    Ehandler.Edict.update(handler.Edict)
                    if mbe_exact:
                        Ehandler.Eexact.update(handler.Eexact)
            else:
                tt1 = time.time()
                mpi.comm.send(Ehandler, dest=0, tag=0)
                tt2 = time.time()
                if tt2 - tt1 > 1:
                    print(f"I ({mpi.rank}) sended {time.time()-tt1}.")
            Ehandler = mpi.comm.bcast(Ehandler, root=0)
            Ehandler = mbe_comm.bcast(Ehandler, root=0)
            return

        def Compute_E_Ehandler(Ehandler, core, sec):
            ##################
            # Compute energy #
            ##################
            Embe = None
            Embe_exact = None
            if not Quket.later and mpi.handler_rank:
                Ehandler.load(core, sec)
                Ehandler.convert2delta(core, sec)
                Embe = sum(Ehandler.Edict.values())
                if mbe_exact:
                    Embe_exact = sum(Ehandler.Eexact.values())
                Ehandler.save_current(core, sec, ext="delta")
            Embe = mpi.comm.bcast(Embe, root=0)
            Embe_exact = mpi.comm.bcast(Embe_exact, root=0)
            return Embe, Embe_exact

        def wrap_mbe_energy(Ehandler, core, sec):
            #######################
            # Compute Energy ...  #
            #######################
            mpi.comm = mpi.MPI.COMM_WORLD
            mpi.rank = mpi.comm.Get_rank()
            mpi.nprocs = mpi.comm.Get_size()
            mpi.handler_rank = 1 if mpi.rank == 0 else 0
            cf._log = cf.log

            Send_Receive_Ehandler(Ehandler)
            if mpi.handler_rank:
                Ehandler.save_current(core, sec)
    
            return Compute_E_Ehandler(Ehandler, core, sec)




        if Quket.method != "mbe":
            # Do VQE.
            return vqe(Quket, *args, **kwds)

        t1 = time.time()

        noa = Quket.noa
        nob = Quket.nob
        nva = Quket.nva
        nvb = Quket.nvb
        nf = Quket.n_frozen_orbitals
        nc = Quket.n_core_orbitals
        na = Quket.n_active_orbitals
        ### Restore inert n_secondary_orbitals
        Quket.n_secondary_orbitals = Quket._n_secondary_orbitals
        ns = Quket.n_secondary_orbitals
        core_offset = Quket.core_offset
        min_use_core = Quket.min_use_core
        max_use_core = Quket.max_use_core
        min_use_sec = Quket.min_use_secondary
        max_use_sec = Quket.max_use_secondary
        n_orbitals = nc + na + ns
        include = copy(Quket.include)
        color = Quket.color
        det = Quket.det
        maxiter = Quket.maxiter
        mo_basis = Quket.mo_basis
        mbe_exact = Quket.mbe_exact
        mbe_correlator = Quket.mbe_correlator
        ansatz = Quket.ansatz
        taper_off = Quket.cf.do_taper_off
        from_vir = Quket.from_vir

        ### Initialization for MPI related variables ###
        mpi.handler_rank = 0
        my_color = 0
        my_key = 0
        cf._log = cf.log
        ###

        core_list = np.arange(nc)
        sec_list = np.arange(nc+na, nc+na+ns)
        Ehandler = EnergyHandler(f"{Quket.name}_{Quket.mo_basis}",
                                 nc, na, ns, min_use_core, max_use_core,
                                 min_use_sec, max_use_sec, mbe_exact)

        # Trap invalid input.
        if nc < max_use_core:
            error(f"n_core_orbitals={nc} is too small "
                  f"to use {max_use_core} core orbitals.")
        if ns < max_use_sec:
            error(f"n_secondary_orbitals={ns} is too small "
                  f"to use {max_use_sec} virtual orbitals.")
        if max(noa, nob) > na:
            error(f"The electron is in the secondary orbital.",
                  f"max(noa, nob)=({max(noa, nob)} > n_active_electrons={na})")

        ############
        # Get base #
        ############
        if max(noa, nob) == na:
            sig = [param for param in inspect.signature(vqe).parameters]
            idx = sig.index("maxiter") - 1
            maxiter = args[idx]
            args = *args[:idx], -1, *args[idx+1:]
        if Quket.mbe_oo:
            oo_kwds = deepcopy(Quket._kwds)
            oo_kwds["from_vir"] = False
            oo_kwds["method"] = "vqe"
            if Quket.mbe_oo_ansatz is None:
                oo_kwds["ansatz"] = Quket.ansatz
            else:
                oo_kwds["ansatz"] = Quket.mbe_oo_ansatz
            oo_kwds["n_core_orbitals"] = 0
            oo_kwds["n_secondary_orbitals"] = 0
            oo_kwds["include"] = {"a": "a", "aa": "aa"}
            oo_init_kwds = get_func_kwds(QuketData.__init__, oo_kwds)
            oo_Quket = QuketData(**oo_init_kwds)
            set_config(oo_kwds, oo_Quket)
            oo_Quket.initialize(**oo_kwds)
            oo_Quket.openfermion_to_qulacs()
            ### Tweaking orbitals...
            if oo_Quket.alter_pairs != []:
                ## Switch orbitals
                oo_Quket.alter(Quket.alter_pairs)
            oo_Quket.get_pauli_list()
            oo_Quket.set_projection()
            if oo_Quket.cf.do_taper_off:
                oo_Quket.taper_off()
            oo(oo_Quket)
            oo_Quket = set_params(oo_Quket, nc, ns)
            oo_Quket.n_frozen_orbitals = nf
            orbital_rotation(oo_Quket, mbe=True)
            Hamiltonian = oo_Quket.operators.Hamiltonian
            Quket.state = oo_Quket.state.copy()
            Quket.ndim = oo_Quket.ndim
            Quket._ndim = oo_Quket._ndim
            Ebase, S2base = oo_Quket.get_E(), oo_Quket.get_S2()
            base_vec = oo_Quket.state.get_vector()
            del(oo_Quket)

            if mpi.handler_rank:
                norbs = nc + na + ns
                S2 = s_squared_operator(norbs)
                Number = number_operator(norbs*2)
                rx = create_1body_operator(oo_Quket.rint[0], mo_coeff=oo_Quket.mo_coeff,
                                           ao=True, n_active_orbitals=norbs)
                ry = create_1body_operator(oo_Quket.rint[1], mo_coeff=oo_Quket.mo_coeff,
                                           ao=True, n_active_orbitals=norbs)
                rz = create_1body_operator(oo_Quket.rint[2], mo_coeff=oo_Quket.mo_coeff,
                                           ao=True, n_active_orbitals=norbs)
                Dipole = [rx, ry, rz]
            else:
                S2 = None
                Number = None
                Dipole = None
            S2 = mpi.comm.bcast(S2, root=0)
            Number = mpi.comm.bcast(Number, root=0)
            Dipole = mpi.comm.bcast(Dipole, root=0)
            #oo_Quket.operators.S2 = S2
            #oo_Quket.operators.Number = Number
            #oo_Quket.operators.Dipole = Dipole
            #oo_Quket.operators.jordan_wigner()
            #oo_Quket.openfermion_to_qulacs()
        else:
            ##############################
            # Generate whole hamiltonian #
            #   (include core orbitals)  #
            ##############################
            Quket.nc = 0
            Quket.na += nc + ns
            Quket.n_active_electrons += nc*2
            Hamiltonian, S2, Number, Dipole \
                        = Quket.get_operators(guess=Quket.pyscf_guess,
                                              run_fci=False,
                                              run_ccsd=False,
                                              run_mp2=False,
                                              run_casscf=False,
                                              bcast_Hamiltonian=False)
            Quket.two_body_integrals \
                    = Quket.two_body_integrals.transpose(0, 2, 3, 1)
            Quket.nc = nc
            Quket.na -= nc + ns
            Quket.n_active_electrons -= nc*2
#        from pympler import asizeof
#        prints('Size of two_body_integrals_active: ',mpi.rank, asizeof.asizeof(Quket.two_body_integrals_active), root=mpi.rank)
#        prints('Size of Hamiltonian: ', mpi.rank, asizeof.asizeof(Hamiltonian), root=mpi.rank)
        Quket.include = {"a": "a", "aa": "aa"}
        Quket = set_params(Quket, 0, 0)
        if mpi.rank == 0:
            effective_hamiltonian, orbital_indices = get_effective_hamiltonian(Hamiltonian,
                                                          core_list,
                                                          (),
                                                          n_orbitals)
        else:
            effective_hamiltonian, orbital_indices = None, None
        effective_hamiltonian = mpi.bcast(effective_hamiltonian)
        orbital_indices = mpi.bcast(orbital_indices)

        Quket = set_operators(Quket, effective_hamiltonian, core_offset, orbital_indices)
        Quket.get_pauli_list()
        if not Quket.mbe_oo:
            if Quket.cf.do_taper_off:
                Quket.taper_off()
            Ebase, S2base = vqe(Quket, *args, **kwds)
            if Quket.cf.do_taper_off:
                Quket.taper_off(backtransform=True)
            base_vec = Quket.state.get_vector()
        if max(noa, nob) == na:
            args = *args[:idx], maxiter, *args[idx+1:]
        Ehandler[((), ())] = Ebase
        if mbe_exact:
            if ((), ()) not in Ehandler.Eexact:
                Ehandler.Eexact[((), ())] \
                        = eigenspectrum(Quket.operators.Hamiltonian)[0]
        Quket.include = include
        Quket.ansatz = mbe_correlator
        ### Ration of allowed pauli strings.
        ratio = Quket.ndim/Quket._ndim
        file_exist = False
# theta_listを読み込んで利用するオプションを作成する方が実際に即している
        Quket.print_state()
        prints(f"\n===================================================")
        prints(f"  Core = 0   Secondary = 0  (Reference)")
        prints(f"  E[MBE-{Quket.ansatz.upper()}] = {Ebase:.12f}")
        prints(f"====================================================\n")

        ############################
        # Compute auxiliary energy #
        ############################
        Embe = {}
        Embe_exact = {}
        nprocs = mpi.nprocs
        for core in range(min_use_core, max_use_core+1):
            for core_tup in combinations(core_list, core):
                if mpi.rank == 0:
                    effective_hamiltonian, orbital_indices = get_effective_hamiltonian(Hamiltonian,
                                                                  core_list,
                                                                  core_tup,
                                                                  n_orbitals)
                else:
                    effective_hamiltonian, orbital_indices = None, None
                effective_hamiltonian = mpi.bcast(effective_hamiltonian)
                orbital_indices = mpi.bcast(orbital_indices)

                for sec in range(min_use_sec, max_use_sec+1):
                    if core == sec == 0:
                        continue
                    t_0 = time.time()
                    Quket = set_params(Quket, core, sec)
                    sec_com = list(combinations(sec_list, sec))

                    ##########################
                    # Split MPI COMMUNICATOR #
                    ##########################
                    mbe_comm = mpi.MPI.COMM_WORLD
                    mbe_rank = mbe_comm.Get_rank()
                    ### Estimate based on SA-UCCSD 
                    if from_vir:
                        ndim1 = na * (sec + core)
                        ndim2 = na * (na+1) * (sec *(sec+1)//2 + core*(core+1)//2) \
                              + na * (na+1) * na * (sec + core) \
                              + core * na* na* sec \
                              + core*(core-1) * na * sec \
                              + core*(core-1) * sec*(sec-1)//2
                        ndim = ndim1 + ndim2
                    else:
                        _, _, ndim = get_ndims(Quket)
                    ndim = int(ndim * ratio)
                    if color is None:
                        param_per_core = 16
                        core_per_comm = math.ceil(ndim/param_per_core)
                        n_color = math.ceil(nprocs/core_per_comm)

                        #ncalc_nprocs = len(sec_com)//nprocs
                        #ncalc_nprocs_mod = len(sec_com)%nprocs
                        #nprocs_ncalc = nprocs//len(sec_com)
                        #nprocs_ncalc_mod = nprocs % len(sec_com)
                        #ndim_nprocs = ndim//nprocs
                        #ndim_nprocs_mod = ndim%nprocs
                        #if ncalc_nprocs > 3:
                        #    # Many combinations
                        #    if ncalc_nprocs_mod < nprocs//3:
                        #        ### Maybe simply devide by nprocs
                        #        n_color = nprocs
                        #    else:
                        #        n_color = max(math.ceil(np.sqrt(nprocs)), 1) 
                        #elif ncalc_nprocs == 0:
                        #    # Few combinations
                        #    if nprocs_ncalc_mod < nprocs//3: 
                        #        n_color = len(sec_com)
                        #    else:
                        #        #n_color = max(len(sec_com)//2, 1)
                        #        n_color = max(math.ceil(np.sqrt(nprocs)), 1) 
                                
#                        elif ncalc_nprocs > 0:
#                            if ncalc_nprocs_mod < nprocs//5:
#                                n_color = len(sec_com)
#                                print('cccc')
#                            else:
#                                n_color = len(sec_com)//2
#                                print('dddd')
#                        else:      
#                            param_per_core = 16
#                            core_per_comm = math.ceil(ndim/param_per_core)
#                            n_color = math.ceil(nprocs/core_per_comm)
#                            print('eeee')
#                            #if ncalc_nprocs_mod < nprocs//5:
#                            #    n_color = len(sec_com)
#                            #else:
#                            #    n_color = len(sec_com)//2
#                            #n_color = min(len(sec_com), nprocs)

                    elif color < 0:
                        param_per_core = ndim
                        core_per_comm = - color
                        n_color = math.ceil(nprocs/core_per_comm)
                    else:
                        core_per_comm = math.ceil(nprocs/color)
                        param_per_core = math.ceil(ndim/core_per_comm)
                        n_color = color
                    if n_color > len(sec_com):
                        n_color = len(sec_com)
                    my_color = mbe_rank%n_color
                    my_key = mbe_rank//n_color
                    if cf.debug:
                        print(f"I am rank {mbe_comm.rank} "
                              f"assigned (my_color, my_key) = {my_color, my_key}")
                    mpi.comm = mbe_comm.Split(my_color, my_key)
                    mpi.rank = mpi.comm.Get_rank()
                    mpi.nprocs = mpi.comm.Get_size()
                    if my_key == 0 and my_color == 0 :
                        prints(f"We have {len(sec_com)} calculations forward, and {nprocs} processors for MPI.")
                        prints(f"They are split to {n_color} jobs, each has about {nprocs//n_color} procs.")
                    mpi.handler_rank = 1 if my_key == 0 else 0
                    mpi.main_rank = 0
                    cf._log = cf.log
                    if mpi.rank == 0:
                        mpi.main_rank = 1
                        cf.log = f"{cf.input_dir}/{cf.log_file}_MBE_{my_color}.out"
                        if file_exist:
                            prints(f"Using {mpi.nprocs} processors", opentype="a")
                        else:
                            prints(f"Using {mpi.nprocs} processors", opentype="w")
                            file_exist = True

                    my_ndim = math.ceil(len(sec_com)/n_color)
                    ipos = my_color*my_ndim
                    for sec_tup in sec_com[ipos:ipos+my_ndim]:
# set_operatorsでtheta_list指定された場合はそれをreadして再度VQEを流し、base_stateを生成する方が実際的
                        Quket = set_operators(Quket,
                                              effective_hamiltonian,
                                              core_offset,
                                              orbital_indices,
                                              base_vec=base_vec,
                                              core_tup=core_tup,
                                              sec_tup=sec_tup)
                        Quket.get_pauli_list()
                        if Quket.cf.do_taper_off:
                            Quket.tapering.initialized = 0
                            Quket.taper_off()
                        Eaux, S2aux = vqe(Quket, *args, **kwds)
                        if Quket.cf.do_taper_off:
                            Quket.taper_off(backtransform=True)

                        Ehandler[(core_tup, sec_tup)] = Eaux
                        if mbe_exact:
                            if (core_tup, sec_tup) not in Ehandler.Eexact:
                                Ehandler.Eexact[(core_tup, sec_tup)] \
                                        = eigenspectrum(
                                                Quket.operators.Hamiltonian)[0]
                        str_core = f"{core_tup},"
                        str_sec = f"{sec_tup}"
                        len_core = str((core+1) * 3)
                        len_sec = str((sec+1) * 3)
                        prints(f"{str_core:{len_core}} {str_sec:{len_sec}}: E = {Eaux:.12f}  "
                               f"<S**2> = {S2aux:+17.15f}"
                               f"  (DE = {Ehandler.calc_current_delta((core_tup, sec_tup)):+.12f})", filepath=cf._log)
                    mbe_comm.barrier()
                    # Retrieve cf.log
                    cf.log = cf._log
                    mpi.comm.Free()
                    if core_tup == list(combinations(core_list, core))[-1]:
                        Embe[(core, sec)], Embe_exact[(core,sec)] = (wrap_mbe_energy(Ehandler, core, sec))
                        Ehandler = mbe_comm.bcast(Ehandler, root=0)
                        t_1 = time.time()

                        if mpi.handler_rank:
                            prints(f"===================================================", filepath=cf._log)
                            prints(f"  Core = {core}   Secondary = {sec} ", filepath=cf._log)
                            prints(f"  E[MBE-{Quket.ansatz.upper()}] = {Embe[(core,sec)]:.12f}", filepath=cf._log)
                            if mbe_exact:
                                prints(f"  E[MBE-{Quket.ansatz.upper()}-EXACT] = {Embe_exact[(core,sec)]:.12f}",filepath=cf._log)
                            prints(f"  Timing = {t_1-t_0:.2f} sec")
                            prints(f"====================================================\n", filepath=cf._log)

        #Embe, Embe_exact = wrap_mbe_energy(Ehandler, max_use_core, max_use_sec)


        t2 = time.time()
        cput = t2 - t1
        if mpi.handler_rank:
            prints(f"\nMBE Done: CPU Time = {cput:15.4f}\n", filepath=cf._log)

        #if my_color == my_key == 0:
        #   mpi.main_rank = 1
        #else:
        #   mpi.main_rank = 0
        cf.log = cf._log
        mpi.comm = mpi.MPI.COMM_WORLD
        mpi.rank = mpi.comm.Get_rank()
        mpi.nprocs = mpi.comm.Get_size()
        if mpi.rank == 0:
            mpi.main_rank = 1
        else:
            mpi.main_rank = 0
        return Embe, S2


    return wrapper


def set_params(Quket, nc, ns):
    norbs = nc + Quket.na + ns
    Quket.n_qubits = norbs*2
    Quket.nc = nc
    Quket.ns = ns
    Quket.n_frozenv_orbitals = Quket.n_orbitals - (Quket.nf + Quket.nc + Quket.na + Quket.ns)
    Quket.energy = 0.0
    Quket.current_det = (Quket.det << nc*2) + (2**(nc*2) - 1)
    return Quket


def set_operators(Quket, base_hamiltonian, core_offset, orbital_indices,
                  base_vec=None, core_tup=None, sec_tup=None):
    """Ready operators and so on for MBE.

    Args:
        Quket (QuketData): Quket data.
        base_hamiltonian (InteractionOperator): Base hamiltonian for
                                                auxiliary computations.
        base_vec (ndarray): Base vector of quantum state.
        core_tup (Tuple): Using core orbitals.
        sec_tup (Tuple): Using secondary orbitals.

    Returns:
        No return.

    Author(s): Yuma Shimomoto
    """
    from quket.opelib import create_1body_operator
    nc = Quket.nc
    na = Quket.na
    ns = Quket.ns
    det = Quket.current_det
    n_qubits = Quket.n_qubits
    norbs = nc + na + ns

    # Initializing for tapering
    Quket.state = QuantumState(n_qubits)
    Quket.state.set_computational_basis(det)
    if Quket.cf.mapping == "bravyi_kitaev":
        Quket.state = transform_state_jw2bk(Quket.state)
    Quket._n_qubits = n_qubits
    if hasattr(Quket, 'theta_list') and Quket.theta_list is not None:
        del(Quket.theta_list)
    Quket.tapered['theta_list'] = False
    Quket.init_state = QuantumState(n_qubits)
    Quket.init_state.set_computational_basis(det)
    if Quket.cf.mapping == "bravyi_kitaev":
        Quket.init_state = transform_state_jw2bk(Quket.init_state)


    if base_vec is not None:
        from quket.utils import append_01qubits
        #Quket.init_state = append_01qubits(Quket.init_state, nc, 0)
        indices = np.arange(base_vec.size)
        indices = (indices << nc*2) + (2**(nc*2) - 1)
        vec = np.zeros(2**n_qubits, complex)
        vec[indices] = base_vec
        Quket.init_state.load(vec)

    prints(f"core_tup = {core_tup}, sec_tup = {sec_tup}")
    if mpi.rank==0:
        Hamiltonian, fancy_indices \
                = create_using_hamiltonian(Quket,
                                           base_hamiltonian,
                                           core_offset,
                                           core_tup=core_tup,
                                           sec_tup=sec_tup)
        S2 = s_squared_operator(norbs)
        Number = number_operator(norbs*2)
        rx = create_1body_operator(Quket.rint[0], mo_coeff=Quket.mo_coeff,
                                   ao=True, n_active_orbitals=norbs)
        ry = create_1body_operator(Quket.rint[1], mo_coeff=Quket.mo_coeff,
                                   ao=True, n_active_orbitals=norbs)
        rz = create_1body_operator(Quket.rint[2], mo_coeff=Quket.mo_coeff,
                                   ao=True, n_active_orbitals=norbs)
        Dipole = [rx, ry, rz]
    else:
        Hamiltonian = None
        S2 = None
        Number = None
        Dipole = None
        fancy_indices = None
    Hamiltonian = mpi.comm.bcast(Hamiltonian, root=0)
    S2 = mpi.comm.bcast(S2, root=0)
    Number = mpi.comm.bcast(Number, root=0)
    Dipole = mpi.comm.bcast(Dipole, root=0)
    fancy_indices = mpi.comm.bcast(fancy_indices, root=0)
    #fancy_indices_ = orbital_indices[Quket.nf*2:Quket.nf*2+na*2]
    fancy_indices_ = orbital_indices[:na*2]
    #prints('Quket.nf', Quket.nf)
    #prints('Quket.nc', Quket.nc)
    #prints('Quket.na', Quket.na, na)
    #prints('orbital_indices',orbital_indices)
    #prints('fancy_indices_',fancy_indices_)
    if not sec_tup is None:
        for sec_orb in sec_tup:
            fancy_indices_.append(sec_orb*2)
            fancy_indices_.append(sec_orb*2 + 1)
    fancy_indices_ = list(map(lambda x: x + Quket.nf * 2, fancy_indices_))
    prints(f"used spin orbital: {fancy_indices_}")

    Quket.operators.Hamiltonian = Hamiltonian
    Quket.operators.S2 = S2
    Quket.operators.Number = Number
    Quket.operators.Dipole = Dipole
    if Quket.cf.mapping == "jordan_wigner":
        Quket.operators.jordan_wigner()
    elif Quket.cf.mapping == "bravyi_kitaev":
        Quket.operators.bravyi_kitaev(Quket.n_qubits)
    Quket.openfermion_to_qulacs()

    if Quket.symmetry and Quket.symm_operations is not None:
        nfrozen = Quket.n_frozen_orbitals * 2
        ncore = Quket.n_core_orbitals * 2
        nact = Quket.n_active_orbitals * 2
        #fancy_indices_ = list(map(lambda x: x+nfrozen, fancy_indices))
        #pgs_head, pgs_tail = nfrozen, nfrozen+ncore+nact
        irrep_list_mod = [x for ind, x in enumerate(Quket.irrep_list)
                          if ind in fancy_indices_]
        character_list_mod = [
                [x0 for ind, x0 in enumerate(x) if ind in fancy_indices_]
                for x in Quket.character_list]
        Quket.operators.pgs = (Quket.symm_operations,
                               irrep_list_mod,
                               character_list_mod)
    else:
        Quket.operators.pgs = None
    if Quket.cf.do_taper_off:
        from quket.tapering import Z2tapering
        Quket.tapering = Z2tapering(Quket.operators.qubit_Hamiltonian,
                                    Quket.n_qubits,
                                    Quket.current_det,
                                    Quket.operators.pgs)
    return Quket


def get_effective_hamiltonian(hamiltonian, core_list, core_tup, n_orbitals):
    """
    Get effective hamiltonian from whole hamiltonian.

    Args:
        hamiltonian (InteractionOperator):
                Whole hamiltonian includes core, active, secondary space.
        core_list (ndarray|[int]): Spatial orbitals in core space.
        core_tup ((int)): Using spatial orbitals in core space.
        n_orbitals (int): Number of whole orbitals.
                          (core, active, secondary spatial orbitals)

    Returns:
        effective_hamiltonian (InteractionOperator): Effective hamiltonian.

    Author(s): Yuma Shimomoto
    """
    core_indices = []
    for core_orb in core_list:
        if core_orb not in core_tup:
            core_indices.append(core_orb*2)
            core_indices.append(core_orb*2 + 1)
    core_indices_2d = np.ix_(core_indices, core_indices)

    effective_indices = []
    for i in range(n_orbitals):
        if i not in core_list or i in core_tup:
            effective_indices.append(i*2)
            effective_indices.append(i*2 + 1)
    effective_indices_2d = np.ix_(effective_indices, effective_indices)
    effective_indices_4d = np.ix_(effective_indices, effective_indices,
                                  effective_indices, effective_indices)
    whole_indices = [i for i in range(2*n_orbitals)]
    whole_indices_2d = np.ix_(whole_indices, whole_indices)
    whole_indices_4d = np.ix_(whole_indices, whole_indices,
                              whole_indices, whole_indices)

    constant = hamiltonian[()]
    one_body_tensor = hamiltonian[(whole_indices_2d[0], 1),
                                  (whole_indices_2d[1], 0)]
    two_body_tensor = hamiltonian[(whole_indices_4d[0], 1),
                                  (whole_indices_4d[1], 1),
                                  (whole_indices_4d[2], 0),
                                  (whole_indices_4d[3], 0)]

    # It should be noted that it is different from the formula
    # because it is calculated by tensor instead of integrals.
    # Also, since the tensor is based on the spin orbital,
    # (ij|ji) and (ip|qi) are doubled.
    effective_constant = 2*np.diag(one_body_tensor[core_indices_2d]).sum()
    for i, j in product(core_indices, repeat=2):
        # Formulation; Ec = 2 \sum_i^nc h_ii + \sum_ij^nc 2(ii|jj) - (ij|ji)
        effective_constant += 2*(two_body_tensor[i, j, j, i]
                               - two_body_tensor[i, j, i, j])

    effective_one_body_tensor = np.copy(one_body_tensor)
    for p, q in product(effective_indices, repeat=2):
        for i in core_indices:
            # Formulation; hc_pq = h_pq + \sum_i^nc 2(ii|pq) - (ip|qi)
            effective_one_body_tensor[p, q] \
                    += 2*(two_body_tensor[i, p, q, i]
                        - two_body_tensor[i, q, i, p])

    effective_hamiltonian = InteractionOperator(
            constant + effective_constant/2,
            effective_one_body_tensor[effective_indices_2d],
            two_body_tensor[effective_indices_4d])
    return effective_hamiltonian, effective_indices 


def create_using_hamiltonian(Quket, hamiltonian, core_offset,
                             core_tup=None, sec_tup=None):
    """
    This function creates hamiltonian to use calculation of energy by
    'Many-Body Expansion' method.

    Args:
        Quket (QuketData): Quket data.
        hamiltonian (InteractionOperator): Hamiltonian applying this method.
        core_tup (Tuple): Using core orbitals.
        sec_tup (Tuple): Using secondray orbitals.

    Returns:
        new_hamiltonian (InteractionOperator): Created hamiltonian.

    Author(s): Yuma Shimomoto
    """
    nc = Quket.nc
    na = Quket.na
    ns = Quket.ns

    ######################
    # Ready of constants #
    ######################
    nfrozen = Quket.nf*2
    n_core_spin_orbitals = nc*2
    n_active_spin_orbitals = na*2
    n_virtual_spin_orbitals = ns*2
    n_spin_orbitals = (n_core_spin_orbitals
                       + n_active_spin_orbitals
                       + n_virtual_spin_orbitals)

    ######################
    # Make fancy indices #
    ######################
    fancy_indices = [i for i in range(n_core_spin_orbitals
                                      + n_active_spin_orbitals)]

    if not sec_tup is None:
        for sec_orb in sec_tup:
            sec_orb -= core_offset - nc
            fancy_indices.append(sec_orb*2)
            fancy_indices.append(sec_orb*2 + 1)
    fancy_indices_2d = np.ix_(fancy_indices, fancy_indices)
    fancy_indices_4d = np.ix_(fancy_indices, fancy_indices,
                              fancy_indices, fancy_indices)

    #####################
    # Allocate memories #
    #####################
    one_body_tensor = np.zeros((n_spin_orbitals,)*2, dtype=float)
    two_body_tensor = np.zeros((n_spin_orbitals,)*4, dtype=float)

    ##############
    # Set values #
    ##############
    constant = hamiltonian[()]
    one_body_tensor = hamiltonian[(fancy_indices_2d[0], 1),
                                  (fancy_indices_2d[1], 0)]
    two_body_tensor = hamiltonian[(fancy_indices_4d[0], 1),
                                  (fancy_indices_4d[1], 1),
                                  (fancy_indices_4d[2], 0),
                                  (fancy_indices_4d[3], 0)]

    ####################################
    # Set variables to new hamiltonian #
    ####################################
    new_hamiltonian = InteractionOperator(constant,
                                          one_body_tensor,
                                          two_body_tensor)
    return new_hamiltonian, fancy_indices


@dataclass
class EnergyHandler():
    fname: str
    nc: int
    na: int
    ns: int
    min_use_core: int
    max_use_core: int
    min_use_sec: int
    max_use_sec: int
    mbe_exact: bool

    Edict: Dict[Tuple, float] = field(init=False, default_factory=dict)
    DEdict: Dict[Tuple, float] = field(init=False, default_factory=dict)
    Eexact: Dict[Tuple, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        if self.mbe_exact:
            self.load(ext="exact")

    def __setitem__(self, key, value):
        self.Edict[key] = value

    def __getitem__(self, key):
        return self.Edict[key]

    def convert2delta(self, core, sec):
        under_core = self.nc
        over_act = self.nc + self.na
        core_list = np.arange(under_core)
        sec_list = np.arange(over_act, over_act+self.ns)
        for core in range(core+1):
            for core_tup in combinations(core_list, core):
                for sec in range(sec+1):
                    for sec_tup in combinations(sec_list, sec):
                        self.calc_delta((core_tup, sec_tup))

    def calc_current_delta(self, key):
        DE = self.Edict[key]
        for i in range(len(key[0])+1):
            for core_sub_tup in combinations(key[0], i):
                for j in range(len(key[1])+1):
                    for sec_sub_tup in combinations(key[1], j):
                        if key == (core_sub_tup, sec_sub_tup):
                            continue
                        DE -= self.Edict[(core_sub_tup, sec_sub_tup)]
        return DE

    def calc_delta(self, key):
        if cf.debug:
            prints(f"Compute (core={key[0]}, secondary={key[1]}) "
                    "set's contribution.")
            prints(f"    My energy: {self.Edict[key]}")
            if self.mbe_exact:
                prints(f"My FCI energy: {self.Eexact[key]}")
                prints(f"  diff energy: {self.Edict[key]-self.Eexact[key]}")
        for i in range(len(key[0])+1):
            for core_sub_tup in combinations(key[0], i):
                for j in range(len(key[1])+1):
                    for sec_sub_tup in combinations(key[1], j):
                        if key == (core_sub_tup, sec_sub_tup):
                            continue

                        if cf.debug:
                            prints(f"    subsets; core={core_sub_tup}, "
                                           f"secondary={sec_sub_tup}")

                        self.Edict[key] \
                                -= self.Edict[(core_sub_tup, sec_sub_tup)]
                        if self.mbe_exact:
                            self.Eexact[key] \
                                    -= self.Eexact[(core_sub_tup, sec_sub_tup)]
        if cf.debug:
            prints(f"Computed contribution energy = {self.Edict[key]}.\n")
            if self.mbe_exact:
                prints(f"Computed contribution (exact) energy "
                       f"= {self.Eexact[key]}.\n")

    def load(self, cur_use_core, cur_use_sec, ext="energy"): 
        for core in range(cur_use_core+1):
            for sec in range(cur_use_sec+1):
                if os.path.exists(f"{cf.input_dir}/{cf.log_file}_MBE_"
                                  f"c{core}a{self.na}s{sec}.{ext}"):
                    with open(f"{cf.input_dir}/{cf.log_file}_MBE_"
                              f"c{core}a{self.na}s{sec}.{ext}",
                              "rb") as f:
                        Edict = pickle.load(f)
                    if ext == "exact":
                        self.Eexact.update(Edict)
                    else:
                        self.Edict.update(Edict)

    def save(self, ext="energy"):
        under_core = self.nc
        over_act = self.nc + self.na
        core_list = np.arange(under_core)
        sec_list = np.arange(over_act, over_act+self.ns)
        for core in range(self.min_use_core, self.max_use_core+1):
            for sec in range(self.min_use_sec, self.max_use_sec+1):
                values = {}
                for core_tup in combinations(core_list, core):
                    for sec_tup in combinations(sec_list, sec):
                        if ext == "exact":
                            values[(core_tup, sec_tup)] \
                                    = self.Eexact[(core_tup, sec_tup)]
                        else:
                            values[(core_tup, sec_tup)] \
                                    = self.Edict[(core_tup, sec_tup)]

                with open(f"{cf.input_dir}/{cf.log_file}_MBE_"
                          f"c{core}a{self.na}s{sec}.{ext}",
                          "wb") as f:
                    pickle.dump(values, f)

    def save_current(self, core, sec,  ext="energy"):
        under_core = self.nc
        over_act = self.nc + self.na
        core_list = np.arange(under_core)
        sec_list = np.arange(over_act, over_act+self.ns)
        values = {}
        for core_tup in combinations(core_list, core): 
            for sec_tup in combinations(sec_list, sec):
                if ext == "exact":
                    values[(core_tup, sec_tup)] \
                            = self.Eexact[(core_tup, sec_tup)]
                else:
                    values[(core_tup, sec_tup)] \
                            = self.Edict[(core_tup, sec_tup)]

        with open(f"{cf.input_dir}/{cf.log_file}_MBE_"
                  f"c{core}a{self.na}s{sec}.{ext}",
                  "wb") as f:
            pickle.dump(values, f)
