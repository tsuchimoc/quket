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
import numpy as np
from itertools import product, combinations, combinations_with_replacement
import itertools
from qulacs import QuantumState
from openfermion.ops import  QubitOperator, FermionOperator
from openfermion.transforms import jordan_wigner,bravyi_kitaev
from openfermion.utils import  hermitian_conjugated

from quket.fileio import prints
from quket.mpilib import mpilib as mpi
from quket.utils import make_gate, int2occ


def convert2List(excite_dict):
    excite_list = []
    singles_list = list(singles(excite_dict))
    doubles_list = list(doubles(excite_dict))
    excite_list.extend(singles_list)
    excite_list.extend(doubles_list)
    return excite_list


def singles(excite_dict):
    for from_ in excite_dict["singles"]:
        a2a, b2b = excite_dict["singles"][from_]

        for i, a in a2a:
            yield i, a
        for i, a in b2b:
            yield i, a

def singles_sf(excite_dict):
    for from_ in excite_dict["singles"]:
        a2a = excite_dict["singles"][from_][0]

        for i, a in a2a:
            yield i, a


def doubles(excite_dict):
    for from_ in excite_dict["doubles"]:
        aa2aa, ab2ab, ba2ba, bb2bb = excite_dict["doubles"][from_]

        for (i, j), (a, b) in aa2aa:
            if b > j:
                yield i, j, a, b
            else:
                yield a, b, i, j
        for (i, j), (a, b) in ab2ab:
            if max(b, a) > max(j, i):
                yield min(j, i), max(j, i), min(b, a), max(b, a)
            else:
                yield min(b, a), max(b, a), min(j, i), max(j, i)
        for (j, i), (b, a) in ba2ba:
            if max(b, a) > max(j, i):
                yield min(j, i), max(j, i), min(b, a), max(b, a)
            else:
                yield min(b, a), max(b, a), min(j, i), max(j, i)
        for (i, j), (a, b) in bb2bb:
            if b > j:
                yield i, j, a, b
            else:
                yield a, b, i, j

def doubles_sf(excite_dict):
    for from_ in excite_dict["doubles"]:
        aa2aa = excite_dict["doubles"][from_][0]

        for (i, j), (a, b) in aa2aa:
            yield i, j, a, b

def get_excite_dict(Quket):
    """Get excitation generator list.

    Args:
        Quket (QuketData): Quket data.

    Returns:
        excite_dict ({"singles": [generators], "doubles": [generators]}):
                Excitation's dictionary.

    Author(s): Yuma Shimomoto
    """
    nc = Quket.nc
    na = Quket.na
    ns = Quket.ns
    # Note; When using core space Since current_det has been changed,
    #       use det to get the quantum state of active space.
    det = Quket.det
    include = Quket.include
    from_vir = Quket.from_vir

    core = [i for i in range(nc*2)]
    if not from_vir:
        occ = int2occ(det)
        occ = [i + nc*2 for i in occ]
        vir = [i for i in range(nc*2, (nc+na)*2) if i not in occ]
    else:
        _occ = int2occ(det)
        _occ = [i + nc*2 for i in _occ]
        _vir = [i for i in range(nc*2, (nc+na)*2) if i not in _occ]
        occ = _occ + _vir
        vir = _occ + _vir
    sec = [i for i in range((nc+na)*2, (nc+na+ns)*2)]

    core_a = core[::2]
    core_b = core[1::2]
    occ_a = [i for i in occ if i%2 == 0]
    occ_b = [i for i in occ if i%2 == 1]
    vir_a = [i for i in vir if i%2 == 0]
    vir_b = [i for i in vir if i%2 == 1]
    sec_a = sec[::2]
    sec_b = sec[1::2]

    excite_dict = {"singles": {}, "doubles": {}}
    for from_, to in include.items():
        # Initialize
        from_list_a = []
        from_list_b = []
        from_list_aa = []
        from_list_ab = []
        from_list_ba = []
        from_list_bb = []
        to_list_a = []
        to_list_b = []
        to_list_aa = []
        to_list_ab = []
        to_list_ba = []
        to_list_bb = []

        # Note; '+' behave 'extend' method in List.
        if from_ == "c":
            from_list_a += core_a
            from_list_b += core_b
            if "a" in to:
                # alpha excitation.
                to_list_a += vir_a
                # beta excitation.
                to_list_b += vir_b
            if "s" in to:
                # alpha excitation.
                to_list_a += sec_a
                # beta excitation.
                to_list_b += sec_b
            a2a = product(from_list_a, to_list_a)
            b2b = product(from_list_b, to_list_b)
            excite_dict["singles"][from_] = a2a, b2b
        elif from_ == "a":
            from_list_a += occ_a
            from_list_b += occ_b
            if "a" in to:
                # alpha excitation.
                to_list_a += vir_a
                # beta excitation.
                to_list_b += vir_b
            if "s" in to:
                # alpha excitation.
                to_list_a += sec_a
                # beta excitation.
                to_list_b += sec_b
            a2a = product(from_list_a, to_list_a)
            b2b = product(from_list_b, to_list_b)
            ### Check the redundancy...
            nonred_list = []
            x_ = []
            for x in (a2a,b2b):
                list_ = list(x) 
                red_list = []
                for i in range(len(list_)):
                    if list_[i][0] == list_[i][1]:
                        red_list.append(i)
                        continue
                    for j in range(i):
                        if (list_[i][0] == list_[j][1] and list_[i][1] == list_[j][0]):
                            #print(' is equivalent to ', list_[j])
                            red_list.append(i)
            
                for m in red_list[::-1]:
                    list_.pop(m)
                x_.append(itertools.chain.from_iterable([list_]))
            excite_dict["singles"][from_] = x_
        elif from_ == "cc":
            from_list_aa += combinations(core_a, 2)
            from_list_ab += product(core_a, core_b)
            from_list_ba += product(core_b, core_a)
            from_list_bb += combinations(core_b, 2)
            if "aa" in to:
                # alpha-alpha excitation.
                to_list_aa += combinations(vir_a, 2)
                # alpha-beta excitation.
                to_list_ab += product(vir_a, vir_b)
                # beta-alpha excitation.
                to_list_ba += []
                # beta-beta excitation.
                to_list_bb += combinations(vir_b, 2)
            if "as" in to:
                # alpha-alpha excitation.
                to_list_aa += product(vir_a, sec_a)
                # alpha-beta excitation.
                to_list_ab += product(vir_a, sec_b)
                # beta-alpha excitation.
                to_list_ba += product(vir_b, sec_a)
                # beta-beta excitation.
                to_list_bb += product(vir_b, sec_b)
            if "ss" in to:
                # alpha-alpha excitation.
                to_list_aa += combinations(sec_a, 2)
                # alpha-beta excitation.
                to_list_ab += product(sec_a, sec_b)
                # beta-alpha excitation.
                to_list_ba += []
                # beta-beta excitation.
                to_list_bb += combinations(sec_b, 2)
            aa2aa = product(from_list_aa, to_list_aa)
            ab2ab = product(from_list_ab, to_list_ab)
            ba2ba = product(from_list_ba, to_list_ba)
            bb2bb = product(from_list_bb, to_list_bb)
            excite_dict["doubles"][from_] = aa2aa, ab2ab, ba2ba, bb2bb
        elif from_ == "ca":
            from_list_aa += product(core_a, occ_a)
            from_list_ab += product(core_a, occ_b)
            from_list_ba += product(core_b, occ_a)
            from_list_bb += product(core_b, occ_b)
            if "aa" in to:
                # alpha-alpha excitation.
                to_list_aa += combinations(vir_a, 2)
                # alpha-beta excitation.
                to_list_ab += product(vir_a, vir_b)
                # beta-alpha excitation.
                to_list_ba += product(vir_b, vir_a)
                # beta-beta excitation.
                to_list_bb += combinations(vir_b, 2)
            if "as" in to:
                # alpha-alpha excitation.
                to_list_aa += product(vir_a, sec_a)
                # alpha-beta excitation.
                to_list_ab += product(vir_a, sec_b)
                to_list_ab += product(sec_a, vir_b)
                # beta-alpha excitation.
                to_list_ba += product(vir_b, sec_a)
                to_list_ba += product(sec_b, vir_a)
                # beta-beta excitation.
                to_list_bb += product(vir_b, sec_b)
            if "ss" in to:
                # alpha-alpha excitation.
                to_list_aa += combinations(sec_a, 2)
                # alpha-beta excitation.
                to_list_ab += product(sec_a, sec_b)
                # beta-alpha excitation.
                to_list_ba += product(sec_a, sec_b)
                # beta-beta excitation.
                to_list_bb += combinations(sec_b, 2)
            aa2aa = product(from_list_aa, to_list_aa)
            ab2ab = product(from_list_ab, to_list_ab)
            ba2ba = product(from_list_ba, to_list_ba)
            bb2bb = product(from_list_bb, to_list_bb)
            excite_dict["doubles"][from_] = aa2aa, ab2ab, ba2ba, bb2bb
        elif from_ == "aa":
            from_list_aa += combinations(occ_a, 2)
            from_list_ab += product(occ_a, occ_b)
            from_list_ba += product(occ_b, occ_a)
            from_list_bb += combinations(occ_b, 2)
            if "aa" in to:
                # alpha-alpha excitation.
                to_list_aa += combinations(vir_a, 2)
                # alpha-beta excitation.
                to_list_ab += product(vir_a, vir_b)
                # beta-alpha excitation.
                to_list_ba += []
                # beta-beta excitation.
                to_list_bb += combinations(vir_b, 2)
            if "as" in to:
                # alpha-alpha excitation.
                to_list_aa += product(vir_a, sec_a)
                # alpha-beta excitation.
                to_list_ab += product(vir_a, sec_b)
                # beta-alpha excitation.
                to_list_ba += product(vir_b, sec_a)
                # beta-beta excitation.
                to_list_bb += product(vir_b, sec_b)
            if "ss" in to:
                # alpha-alpha excitation.
                to_list_aa += combinations(sec_a, 2)
                # alpha-beta excitation.
                to_list_ab += product(sec_a, sec_b)
                # beta-alpha excitation.
                to_list_ba += []
                # beta-beta excitation.
                to_list_bb += combinations(sec_b, 2)
            aa2aa = product(from_list_aa, to_list_aa)
            ab2ab = product(from_list_ab, to_list_ab)
            ba2ba = product(from_list_ba, to_list_ba)
            bb2bb = product(from_list_bb, to_list_bb)

            ### Check the redundancy...
            nonred_list = []
            x_ = []
            for x in (aa2aa, ab2ab, ba2ba, bb2bb):
                list_ = list(x) 
                red_list = []
                for i in range(len(list_)):
                    if list_[i][0] == list_[i][1]:
                        red_list.append(i)
                        continue
                    for j in range(i):
                        if (list_[i][0] == list_[j][1] and list_[i][1] == list_[j][0]):
                            #print(' is equivalent to ', list_[j])
                            red_list.append(i)
            
                for m in red_list[::-1]:
                    list_.pop(m)
                x_.append(itertools.chain.from_iterable([list_]))
            #excite_dict["doubles"][from_] = aa2aa, ab2ab, ba2ba, bb2bb
            excite_dict["doubles"][from_] = x_
    return excite_dict


def get_excite_dict_sf(Quket):
    """Get excitation generator list for spin-free.

    Args:
        Quket (QuketData): Quket data.

    Returns:
        excite_dict ({"singles": [generator], "doubles": [generator]}):
                Excitation's dictionary.

    Author(s): Yuma Shimomoto
    """
    nc = Quket.nc
    na = Quket.na
    ns = Quket.ns
    det = Quket.det
    include = Quket.include
    from_vir = Quket.from_vir

    n_occ = len(int2occ(det))//2
    core = [i for i in range(nc)]
    occ = [i for i in range(nc, nc+n_occ)]
    vir = [i for i in range(nc+n_occ, nc+na)]
    sec = [i for i in range(nc+na, nc+na+ns)]

    if not from_vir:
        occ = [i for i in range(nc, nc+n_occ)]
        vir = [i for i in range(nc+n_occ, nc+na)]
    else:
        _occ = [i for i in range(nc, nc+n_occ)]
        _vir = [i for i in range(nc+n_occ, nc+na)]
        occ = _occ + _vir
        vir = _occ + _vir

    excite_dict = {"singles": {}, "doubles": {}}
    for from_, to in include.items():
        from_list = []
        to_list = []
        if from_ == "c":
            from_list.extend(core)
            if "a" in to:
                to_list.extend(vir)
            if "s" in to:
                to_list.extend(sec)
            excite_dict["singles"][from_] = [product(from_list, to_list)]
        elif from_ == "a":
            from_list.extend(occ)
            if "a" in to:
                to_list.extend(vir)
            if "s" in to:
                to_list.extend(sec)
            ### Check the redundancy...
            nonred_list = []
            x_ = []
            list_ = list(product(from_list, to_list))
            red_list = []
            for i in range(len(list_)):
                if list_[i][0] == list_[i][1]:
                    red_list.append(i)
                    continue
                for j in range(i):
                    if (list_[i][0] == list_[j][1] and list_[i][1] == list_[j][0]):
                        #print(' is equivalent to ', list_[j])
                        red_list.append(i)
            
            for m in red_list[::-1]:
                list_.pop(m)
            x_.append(itertools.chain.from_iterable([list_]))
            #excite_dict["doubles"][from_] = aa2aa, ab2ab, ba2ba, bb2bb
            excite_dict["singles"][from_] = x_
        elif from_ == "cc":
            from_list.extend(product(core, core))
            if "aa" in to:
                ### Note
                ###   from_list takes (aa, ab, bb) types
                ###   to_list takes (aa, ba, bb) types
                ###  [A]  ((3, 4),  (7, 8))
                ###  -->  7A^ 8A^ 4A 3A  + 7B^ 8A^ 4A 3B  + 7A^ 8B^ 4B 3A  + 7B^ 8B^ 4B 3B

                ###  [B]  ((3, 4),  (8, 7))
                ###  -->  8A^ 7A^ 4A 3A  + 8B^ 7A^ 4A 3B  + 8A^ 7B^ 4B 3A  + 8B^ 7B^ 4B 3B

                ###  [C]  ((4, 3),  (7, 8))
                ###  -->  7A^ 8A^ 3A 4A  + 7B^ 8A^ 3A 4B  + 7A^ 8B^ 3B 4A  + 7B^ 8B^ 3B 4B

                ###  [D]  ((4, 3),  (8, 7))
                ###  -->  8A^ 7A^ 3A 4A  + 8B^ 7A^ 3A 4B  + 8A^ 7B^ 3B 4A  + 8B^ 7B^ 3B 4B

                ### [A] = [D] != [B] = [C]
                ### So, from_list takes combinations_with_replacement 
                ### whereas to_list takes product

                to_list.extend(combinations_with_replacement(vir, 2))
            if "as" in to:
                ### ((3, 4), (7, 12))
                ### -->  7A^ 12A^ 4A 3A  + 7A^ 12B^ 4B 3A + 7B^ 12A^ 4A 3B  + 7B^ 12B^ 4B 3B 

                ### ((4, 3), (7, 12))
                ### -->  7A^ 12A^ 3A 4A  + 7A^ 12B^ 3B 4A + 7B^ 12A^ 3A 4B  + 7B^ 12B^ 3B 4B 

                ### So, both from_list and to_list take product
                to_list.extend(product(vir, sec))
            if "ss" in to:
                to_list.extend(combinations_with_replacement(sec, 2))
            excite_dict["doubles"][from_] = [product(from_list, to_list)]
        elif from_ == "ca":
            from_list.extend(product(core, occ))
            if "aa" in to:
                to_list.extend(product(vir, vir))
            if "as" in to:
                to_list.extend(product(vir, sec))
            if "ss" in to:
                to_list.extend(product(sec, sec))
            excite_dict["doubles"][from_] = [product(from_list, to_list)]
        elif from_ == "aa":
            from_list.extend(product(occ, occ))
            if "aa" in to:
                to_list.extend(combinations_with_replacement(vir, 2))
            if "as" in to:
                to_list.extend(product(vir, sec))
            if "ss" in to:
                to_list.extend(combinations_with_replacement(sec, 2))

            ### Check the redundancy...
            nonred_list = []
            x_ = []
            list_ = list(product(from_list, to_list))
            red_list = []
            for i in range(len(list_)):
                if list_[i][0] == list_[i][1]:
                    red_list.append(i)
                    continue
                for j in range(i):
                    if (list_[i][0] == list_[j][1] and list_[i][1] == list_[j][0]):
                        red_list.append(i)
                        break
                    elif list_[i][0][0] == list_[j][1][1] and list_[i][0][1] == list_[j][1][0] and list_[i][1][0] == list_[j][0][1] and list_[i][1][1] == list_[j][0][0]:
                        red_list.append(i)
                        break
                    elif (list_[i][0][0] == list_[i][0][1] == list_[j][0][0] == list_[j][0][1]) and (list_[i][1][0] == list_[j][1][1]) and (list_[i][1][1] == list_[j][1][0]):
                        red_list.append(i)
                        break
                    elif (list_[i][1][0] == list_[i][1][1] == list_[j][1][0] == list_[j][1][1]) and (list_[i][0][0] == list_[j][0][1]) and (list_[i][0][1] == list_[j][0][0]):
                        red_list.append(i)
                        break

            
            for m in red_list[::-1]:
                list_.pop(m)
            x_.append(itertools.chain.from_iterable([list_]))
            #excite_dict["doubles"][from_] = aa2aa, ab2ab, ba2ba, bb2bb
            excite_dict["doubles"][from_] = x_
    return excite_dict


def exciter(wfn, creation_list, annihilation_list):
    '''Function
    Perform the excitation p1! p2! ... pn! qm ... q2 q1 to wfn in Qubit basis.
    n and m do not have to be equal (i.e., ionized states can be obtained).

    Args:
        wfn (QuantumState): Reference wave function to be excited
        creation_list (int list): list of [p1, p2 , ...]
        annihilation_list (int list): list of [q1, q2 , ...]

    Return:
        Excited QuantumState
    '''
    # Qubit counts in wfn
    n_qubits = wfn.get_qubit_count()
    excitation_string = ''
    for p in creation_list:
        excitation_string = excitation_string + str(p)+'^ '
        if p > n_qubits-1:
            raise ValueError(f'Exciter is invoked with p={p} but qubit count in wfn is {n_qubits}.')
    for q in annihilation_list[::-1]:
        excitation_string = excitation_string + str(q)+' '
        if q > n_qubits-1:
            raise ValueError(f'Exciter is invoked with q={q} but qubit count in wfn is {n_qubits}.')
    fermi_op = FermionOperator(excitation_string)
    qubit_op = jordan_wigner(fermi_op)
    return evolve(qubit_op, wfn)


def evolve(operator, wfn, transformation='jordan_wigner', parallel=False):
    """
    Evolve a quantum state by operator,
       wfn' = operator * wfn
    where wfn and wfn' are in the QuantumState class,
    and operator is either in the FermionOperator or QubitOperator class.
    If operator is in the FermionOperator class, first transform it by 'transformation'
    to QubitOperator.
    """
    n_qubits = wfn.get_qubit_count()
    if type(operator) == list:
        for operator_ in operator:
            wfn = evolve(operator_, wfn, transformation=transformation, parallel=parallel)
        return wfn
    elif type(operator) == FermionOperator:
        if transformation == 'jordan_wigner':
            qubit_op = jordan_wigner(operator)
        elif transformation == 'bravyi_kitaev':
            qubit_op = bravyi_kitaev(operator, n_qubits)
    elif type(operator) == QubitOperator:
        qubit_op = operator
    else:
        prints(f'operator = {type(operator)}')
        raise ValueError('operator in evolve() has to be FermionOperator or QubitOperator')
    result = QuantumState(n_qubits)
    result.multiply_coef(0)
    circuit_list = []
    iterm = 0
    for pauli, coef in qubit_op.terms.items():
        if not parallel or iterm % mpi.nprocs == mpi.rank:
            wfn_ = wfn.copy()
            wfn_.multiply_coef(coef)

            index_list = []
            id_list = []
            for k in range(len(pauli)):
                index_list.append(pauli[k][0])
                id_list.append(pauli[k][1])

            circuit = make_gate(n_qubits, index_list, id_list)
            circuit.update_quantum_state(wfn_)

            result.add_state(wfn_)
        iterm += 1

    if parallel:
        # Allreduce to final quantum state
        my_vec = result.get_vector()
        vec = np.zeros_like(my_vec)
        vec = mpi.allreduce(my_vec, mpi.MPI.SUM)
        result.load(vec)
    return result

def FermionOperator_from_list(creation_list, annihilation_list):
    '''Function
    Perform the excitation p1! p2! ... pn! q1 q2 ... qn to wfn in Qubit basis.
    n and m do not have to be equal (i.e., ionized states can be obtained).
    Note: the ordering of annihilation_list is different from that of exciter().

    Args:
        creation_list (int list): list of [p1, p2 , ...]
        annihilation_list (int list): list of [q1, q2 , ...]

    Return:
        Excited QuantumState
    '''
    # Qubit counts in wfn
    excitation_string = ''
    for p in creation_list:
        excitation_string = excitation_string + str(p)+'^ '
    for q in annihilation_list:
        excitation_string = excitation_string + str(q)+' '
    fermi_op = FermionOperator(excitation_string)
    return fermi_op

