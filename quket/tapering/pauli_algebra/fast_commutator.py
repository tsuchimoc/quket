##############
#
#
#
#
##############
"""
FASTER THAN EVER Pauli Operator commutator.
Ground breaking optimization of algorithm and utilization of group theory.
3X to infX speed (depends on the object) compared to ordinary commutator.

Author: TsangSiuChung
"""

import time
from operator import itemgetter
from copy import deepcopy
from itertools import combinations, product, chain

import numpy as np
from openfermion.ops import QubitOperator

from ..tapering import getrid_QOconstant


q2lookup_xyz = {'XY':'Z',
                'YX':'Z',
                'XZ':'Y',
                'ZX':'Y',
                'YZ':'X',
                'ZY':'X'
                }

q2lookup_cof = {'XY':1j,
               'YX':-1j,
               'XZ':-1j,
               'ZX':1j,
               'YZ':1j,
               'ZY':-1j,
               'XX':1,
               'YY':1,
               'ZZ':1
               }

q2lookup_aclw = {'XY':0,
                'YX':1,
                'XZ':1,
                'ZX':0,
                'YZ':0,
                'ZY':1
                }

imag_mul = { (0, 1): -2j,
             (0, 3): 2j,
             (1, 0): 2j,
             (1, 2): 2j,
             (2, 1): 2j,
             (2, 3): -2j,
             (3, 0): -2j,
             (3, 2): 2j }


def imag_mul_gen():
    imag_mul = dict()

    for i,row in enumerate(hollow):
        for j, ety in enumerate(row):
            if ety:
                print(f"({i},{j}) : {ety},")
                imag_mul[(i,j)] = ety
    return imag_mul

    
def how_to_compute_imag_parity(n=4, verbose=True):
    import numpy as np
    from itertools import product
    
    hollow = np.zeros((n,n)).tolist()
    
    parity_set = set()
    parity_set3 = set()
    
    print(' ______________________________________________________________________________________________')    
    print(f"|  clw  aclw   original equation   a=clw%2-aclw%2   b=(clw%4+aclw%4)%4   (a+b)%4  finalresult  |")
    print('|______________________________________________________________________________________________|')
    for clw,aclw in product(range(n), repeat=2):
        
        parity = ((-1)**((aclw)%2) * 1j**((clw + aclw)%4) - (-1)**((clw)%2) * 1j**((clw + aclw)%4))
        
        parity_set.add((  (aclw%2^clw%2), parity)  )

        if aclw%2^clw%2:
            
            ai = (-1)**(aclw%2) - (-1)**(clw%2)
            bi = 1j**(clw%4+aclw%4)
            
            a2i =  aclw%2 - clw%2
            b2i = (clw%4 + aclw%4)%4
            
            ci = ((aclw%2 - clw%2) + (clw%4+aclw%4)%4) %4
            
            parity_set3.add((ci, parity))
            hollow[clw][aclw] = parity

            if verbose:
                idx = f"{clw:>3} {aclw:3}"
                
                
                if aclw%2-clw%2:
                    if not parity: print(f"this is strange.")
                    print(f"|{idx:>8} {ai:>8} * {bi:>8} {a2i:>10} {b2i:>18} {ci:>15} {parity:>12}       |")
    print('|______________________________________________________________________________________________|')                    
    print('\n                                aclw%2^clw%2 = ', parity_set)
    print('(( aclw%2 -clw%2) + (clw%4 + aclw%4)%4)%4 = ', parity_set3)
    
    return np.array(hollow)
        

def fast_commutator_special(A, B):
    """ Version: Special commutator([multiple terms], [single terms])
    
    Find whether A and B is commute i.e. AB == BA or not 
    (In special case)
    Since we need only one counter example to disprove commutativity.
    First assume that it is True.
    When non commute pattern found, break or return False.
    (In general case)
    Since uniqueness is degenerated in the general case, we can not reject
    and commutativity until we reach the end of all calculation.
    Thats why the result is only returned at the end of the function.

    """

    if not isinstance(A, QubitOperator) \
    and not isinstance(B, QubitOperator):
        return None

    qlookup_cof = q2lookup_cof
    
    A_hashs = (dict(k) for k in getrid_QOconstant(A.terms).keys())

    for bkey in B.terms.keys():
        bkey, bidx = dict(bkey), set(map(itemgetter(0), bkey))
        for akey in A_hashs:
            intersect = bidx.intersection(akey.keys())
            if intersect:
                ab, ba = 1, 1
                for i in intersect:
                    if akey[i]!=bkey[i]:
                        ab *= qlookup_cof[f"{akey[i]}{bkey[i]}"]
                        ba *= qlookup_cof[f"{bkey[i]}{akey[i]}"] 
                if ab - ba: 
                    return False
    return True


def fast_commutator_general(A, B, mode=QubitOperator):
    """ Version: General commutator([multiple terms], [multiple terms])
    
    This version do not make use of set theory and relatively slow to 
    self-commute situation however still 100X speed as OpenFermion.

    Find whether A and B is commute i.e. AB == BA or not 
    (In special case)
    Since we need only one counter example to disprove commutativity.
    First assume that it is True.
    When non commute pattern found, break or return False.
    (In general case)
    Since uniqueness is degenerated in the general case, we can not reject
    and commutativity until we reach the end of all calculation.
    Thats why the result is only returned at the end of the function.

    """

    if not isinstance(A, QubitOperator) \
    and not isinstance(B, QubitOperator):
        return None
        

    qlookup_xyz = q2lookup_xyz
    qlookup_cof = q2lookup_cof

    A_hashs = {hash((k,v)):[k,v] for k,v in getrid_QOconstant(A.terms).items()} 
    B_hashs = {hash((k,v)):[dict(k),v] for k,v in getrid_QOconstant(B.terms).items()}
    # Since B.values() is looped for many times, better to batch process
    
    AB_common_terms = set(A_hashs.keys()).intersection(B_hashs.keys())

    new = dict()
    
    for a_hash,(akey,acof) in A_hashs.items():
        akey, aidx = dict(akey), set(map(itemgetter(0), akey))
        if a_hash in AB_common_terms and acof not in {-1,-2}:  # Need to check for b_hash in AB_common_terms

            for b_hash, (bkey,bcof) in B_hashs.items():
            
                if b_hash in AB_common_terms and bcof not in {-1,-2}: 
                    continue
                intersect = aidx.intersection(bkey.keys())
                if intersect:
                    new_key = akey.copy(); new_key.update(bkey)
                    ab, ba = 1, 1
                    
                    for i in intersect:
                        if akey[i]!=bkey[i]:
                            ab *= qlookup_cof[f"{akey[i]}{bkey[i]}"]
                            ba *= qlookup_cof[f"{bkey[i]}{akey[i]}"]
                            new_key[i] = qlookup_xyz[f"{bkey[i]}{akey[i]}"]
                        else:
                            new_key[i] = None

                    new_cof = ab - ba
                    if new_cof:
                        new_key = tuple(sorted(
                                        ((bit,xyz) for bit,xyz in new_key.items() if xyz), key=itemgetter(0)
                                        ))
                        new[new_key] = new.get(new_key, 0) + acof*bcof*new_cof
                        
        else:  # No need to check for b_hash in AB_common_terms, save some times
            for bkey,bcof in B_hashs.values():  # b_hash is not used anywhere below, so .values() is fine
                intersect = aidx.intersection(bkey.keys())
                if intersect:
                    new_key = akey.copy(); new_key.update(bkey)
                    ab, ba = 1, 1
                    for i in intersect:
                        if akey[i]!=bkey[i]:
                            ab *= qlookup_cof[f"{akey[i]}{bkey[i]}"]
                            ba *= qlookup_cof[f"{bkey[i]}{akey[i]}"]
                            new_key[i] = qlookup_xyz[f"{bkey[i]}{akey[i]}"]
                        else:
                            new_key[i] = None
                    new_cof = ab - ba
                    if new_cof:
                        new_key = tuple(sorted(
                                        ((bit,xyz) for bit,xyz in new_key.items() if xyz), key=itemgetter(0)
                                        ))
                        new[new_key] = new.get(new_key, 0) + acof*bcof*new_cof

    if mode == QubitOperator:
        QO = QubitOperator
        new_QO = QO()
        for k,v in new.items():
            if v: new_QO += QO(k,v)
        new_QO.compress()
        return new_QO

    elif mode == dict:
        new = {k:v for k,v in new.items() if val}
        return new
    
    
def fast_commutator_special_parity(A, B):
    """ Version: Special commutator([multiple terms], [single terms])
    
    Find whether A and B is commute i.e. AB == BA or not 
    (In special case)
    Since we need only one counter example to disprove commutativity.
    First assume that it is True.
    When non commute pattern found, break or return False.
    (In general case)
    Since uniqueness is degenerated in the general case, we can not reject
    and commutativity until we reach the end of all calculation.
    Thats why the result is only returned at the end of the function.
    
    Using parity check.

    """

    if not isinstance(A, QubitOperator) \
    and not isinstance(B, QubitOperator):
        return None

    qlookup_aclw = q2lookup_aclw

    A_hashs = (dict(k) for k in getrid_QOconstant(A.terms).keys())

    for bkey in B.terms.keys():
        bkey, bidx = dict(bkey), set(map(itemgetter(0), bkey))
        for akey in A_hashs:
            intersect = bidx.intersection(akey.keys())
            if intersect:
                clw = aclw = 0
                for i in intersect:
                    if akey[i]!=bkey[i]:
                        if qlookup_aclw[f"{akey[i]}{bkey[i]}"]: 
                            aclw ^= 1
                        else: 
                            clw ^= 1
                if aclw^clw: 
                    #print(tuple(akey.items()))
                    #print(tuple(bkey.items()))
                    return False
    return True


def bruteforce_tapering(A, eletroparity=0):
    """ Version: bruteforce tapering-off 
    
    Powered by fast_commutator_special_parity
    
    Use this to find all full set of commute operator of the 
    given QubitOperator A.
    
    """

    if not isinstance(A, QubitOperator):
        return None

    qlookup_aclw = q2lookup_aclw
    t0 = time.time()
    n = max(set( itemgetter(0)(k[-1]) for k in A.terms.keys() if k )) +1
    if eletroparity:
        def callb(n):
            for zn in range(n):
                for ii in combinations(range(n), zn):
                    yield {x:'Z' for x in ii}, ii
        B = callb(n)
        next(B)
    else:
        def callb(n):
            for zn in range(n-2):
                for ii in combinations(range(2,n), zn):
                    yield {x:'Z' for x in ii}, ii
        B = callb(n)
        next(B)        
    
    A_hashs = {k:[dict(k), set(map(itemgetter(0),k))] for k in getrid_QOconstant(A.terms).keys()}
    
    win = []
    for bkey, bidx in B:
        for akey, aidx in A_hashs.values():
            intersect = aidx.intersection(bidx)
            if intersect:
                clw = aclw = 0
                for i in intersect:
                    if akey[i]!=bkey[i]:
                        if qlookup_aclw[f"{akey[i]}{bkey[i]}"]: 
                            aclw ^= 1
                        else: 
                            clw ^= 1
                if aclw^clw: 
                    break
        else:
            win.append(bidx)
    print(f"Done in {(time.time()-t0)} s")
    print('\n'.join(str(x) for x in win))
    
    
def fast_commutator_general_parity_lagacy1(A, B, mode=QubitOperator, debug=0):
    """ Version: General commutator([multiple terms], [multiple terms])
    
    This version do not make use of set theory and relatively slow to 
    self-commute situation however still 100X speed as OpenFermion.

    Find whether A and B is commute i.e. AB == BA or not 
    (In special case)
    Since we need only one counter example to disprove commutativity.
    First assume that it is True.
    When non commute pattern found, break or return False.
    (In general case)
    Since uniqueness is degenerated in the general case, we can not reject
    and commutativity until we reach the end of all calculation.
    Thats why the result is only returned at the end of the function.
    
    Using parity check.

    """
    def body(intersect, akey, acof, bkey, bcof, debug=1):
        new_key = akey.copy(); new_key.update(bkey)
        clw = aclw = 0
        for i in intersect: 
            if akey[i]!=bkey[i]:
                new_key[i] = qlookup_xyz[f"{bkey[i]}{akey[i]}"]
                if qlookup_aclw[f"{akey[i]}{bkey[i]}"]: 
                    aclw += 1  # 5.Utilize of modular and binary arithmetic
                else: clw += 1  # 5.Utilize of modular and binary arithmetic
                
            else:
                new_key[i] = None     

        if aclw%2 ^ clw%2:
            new_key = tuple(sorted(
                            ((bit,xyz) for bit,xyz in new_key.items() if xyz), key=itemgetter(0)
                            ))
            if ((aclw%2-clw%2) + (clw+aclw)%4)%4:
                if new_key in new:
                    new[new_key] -= acof*bcof 
                else: new[new_key] = -acof*bcof 
            else: 
                if new_key in new:
                    new[new_key] += acof*bcof 
                else: new[new_key] = acof*bcof   
                
        #if debug:
        #    aaa = QubitOperator(tuple(akey.items()), acof)
        #    bbb = QubitOperator(tuple(bkey.items()), bcof)
        #    comm = commutator(aaa, bbb)
        #   
        #                
        #    if len(comm.terms):
        #        zeros = [' ' for x in range(14)]
        #        build = [zeros[:], zeros[:]]
        #        for iii, D in enumerate((akey.items(), bkey.items())):
        #            for k,v in D:
        #                build[iii][k] = v
        #        print(np.array(build))
        #    
        #        key, val = list(comm.terms.items())[0]
        #        if key != new_key: print('wrong key')
        #        if val != new_cof: print('wrong val', acof, bcof, clw, aclw)
        #        print(comm)    
        #                        
        #        
        #    if not aclw%2^clw%2 and (len(comm.terms)):
        #        print(new_key, new_cof, "weird")
        #
        #        
        #    if aclw%2 ^ clw%2 and len(comm.terms):
        #        print('True' if new_cof == val else 'False')


    if not isinstance(A, QubitOperator) \
    and not isinstance(B, QubitOperator):
        return None
        
    qlookup_xyz = q2lookup_xyz
    qlookup_aclw = q2lookup_aclw

    A_hashs = {hash((k,v)):[k,v] for k,v in getrid_QOconstant(A.terms).items()} 
    B_hashs = {hash((k,v)):[dict(k),v] for k,v in getrid_QOconstant(B.terms).items()}
    # Since B.values() is looped for many times, better to batch process
    
    AB_common_terms = set(A_hashs.keys()).intersection(B_hashs.keys())

    new = dict()
    
    for a_hash,(akey,acof) in A_hashs.items():
        akey, aidx = dict(akey), set(map(itemgetter(0), akey))
        if a_hash in AB_common_terms and acof not in {-1,-2}:  # Need to check for b_hash in AB_common_terms

            for b_hash, (bkey,bcof) in B_hashs.items():
            
                if b_hash in AB_common_terms and bcof not in {-1,-2}: 
                    continue
                intersect = aidx.intersection(bkey.keys())
                if intersect:
                    body(akey, acof, bkey, bcof, debug)       

        else:  # No need to check for b_hash in AB_common_terms, save some times
            for bkey,bcof in B_hashs.values():  # b_hash is not used anywhere below, so .values() is fine
                intersect = aidx.intersection(bkey.keys())
                if intersect:
                    body(intersect, akey, acof, bkey, bcof, debug)
                            
    if mode == QubitOperator:
        QO = QubitOperator
        new_QO = QO()
        for k,v in new.items():
            if v: new_QO += QO(k,v*2j)
        new_QO.compress()
        return new_QO

    elif mode == dict: 
        new = {k:v*2j for k,v in new.items() if v}
        return new
    
    
def fast_commutator_general_parity_lagacy2(A, B, mode=QubitOperator, debug=0):
    """ Version: General commutator([multiple terms], [multiple terms])
    
    This version make use of set theory and is the fastest solution to 
    self-commute situation while having the same speed as 
    fast_commutator_general_parity.

    Find whether A and B is commute i.e. AB == BA or not 
    (In special case)
    Since we need only one counter example to disprove commutativity.
    First assume that it is True.
    When non commute pattern found, break or return False.
    (In general case)
    Since uniqueness is degenerated in the general case, we can not reject
    and commutativity until we reach the end of all calculation.
    Thats why the result is only returned at the end of the function.
    
    Using parity check.

    """
    def body(intersect, akey, acof, bcof, bkey, debug=0):
        new_key = akey.copy(); new_key.update(bkey)
        clw = aclw = 0
        for i in intersect: 
            if akey[i]!=bkey[i]:
                new_key[i] = qlookup_xyz[f"{bkey[i]}{akey[i]}"]
                if qlookup_aclw[f"{akey[i]}{bkey[i]}"]: 
                    aclw += 1
                else: 
                    clw += 1
                
            else:
                new_key[i] = None   
                
        if aclw%2 ^ clw%2:
            new_key = tuple(sorted(
                            ((bit,xyz) for bit,xyz in new_key.items() if xyz), key=itemgetter(0)
                            ))
            if ((aclw%2-clw%2) + (clw+aclw)%4)%4:
                if new_key in new:
                    new[new_key] -= acof*bcof 
                else: new[new_key] = -acof*bcof 
            else: 
                if new_key in new:
                    new[new_key] += acof*bcof 
                else: new[new_key] = acof*bcof 
                    
        #if debug:
        #    aaa = QubitOperator(tuple(akey.items()), acof)
        #    bbb = QubitOperator(tuple(bkey.items()), bcof)
        #    comm = commutator(aaa, bbb)
        #    if len(comm.terms):
        #        zeros = [' ' for x in range(14)]
        #        build = [zeros[:], zeros[:]]
        #        for iii, D in enumerate((akey.items(), bkey.items())):
        #            for k,v in D:
        #                build[iii][k] = v
        #        print(np.array(build))
        #        key, val = list(comm.terms.items())[0]
        #        if key != new_key: print('wrong key')
        #        if val != new_cof: print('wrong val', acof, bcof, clw, aclw)
        #        print(comm)    
        #    if not aclw%2^clw%2 and (len(comm.terms)):
        #        print(new_key, new_cof, "weird")
        #    if aclw%2 ^ clw%2 and len(comm.terms):
        #        print('True' if new_cof == val else 'False')


    if not isinstance(A, QubitOperator) \
    and not isinstance(B, QubitOperator):
        return None
        
    qlookup_xyz, qlookup_aclw = q2lookup_xyz, q2lookup_aclw
    new = dict()
    A, B = getrid_QOconstant(A.terms), getrid_QOconstant(B.terms)
    
    ABoverlap = set( a for a in 
                    set(A.keys()).intersection(B.keys()) 
                    if A[a]==B[a] 
                    )
    AdifferentB, BdifferentA = set(A.keys())-ABoverlap, set(B.keys())-ABoverlap
    
    B_hashs = tuple([dict(k),B[k]] for k in BdifferentA)
    Boverlap = tuple([dict(k),B[k]] for k in ABoverlap)

    for akey in ABoverlap:
        akey, aidx, acof = dict(akey), set(map(itemgetter(0), akey)), A[akey]
        for (bkey,bcof) in B_hashs:
            intersect = aidx.intersection(bkey.keys())
            if intersect:
                body(intersect, akey, acof, bcof, bkey, debug)
            
    for akey in AdifferentB:
        akey, aidx, acof = dict(akey), set(map(itemgetter(0), akey)), A[akey]
        for (bkey,bcof) in B_hashs:
            intersect = aidx.intersection(bkey.keys())
            if intersect:
                body(intersect, akey, acof, bcof, bkey, debug)  
        for (bkey,bcof) in Boverlap:
            intersect = aidx.intersection(bkey.keys())
            if intersect:
                body(intersect, akey, acof, bcof, bkey, debug)

    if mode == QubitOperator:
        QO = QubitOperator
        new_QO = QO()
        for k,v in new.items():
            if v: new_QO += QO(k,v*2j)
        new_QO.compress()
        return new_QO

    elif mode == dict: 
        new = {k:v*2j for k,v in new.items() if v}
        return new
    
    elif mode == bool:
        for v in set(new.values()):
            if v: return False
    return True   

    
def binary_pauli(H):
    
    if isinstance(H, QubitOperator):
        return {tuple( (b,1) if x=='X' else \
                       (b,2) if x=='Y' else \
                       (b,3) if x=='Z' else x for b,x in k ) : v
                for k,v in H.terms.items() 
                }
            
            
    elif isinstance(H, dict):
        QO = QubitOperator
        new = QO()
        if () in H:
            new += QO((), H[()])
            del H[()]

        for k,v in H.items():
            new += QO(
                      tuple(
                             (b,'X') if x==1 else \
                             (b,'Y') if x==2 else \
                             (b,'Z') if x==3 else x for b,x in k
                            ), v)
        new.compress()
        return new
        
        
def fast_commutator_general_parity(A, B, mode=QubitOperator, debug=0):
    """ Version: General commutator([multiple terms], [multiple terms])
    
    This version make use of set theory and is the fastest solution to 
    self-commute situation while having the same speed as 
    fast_commutator_general_parity.

    Find whether A and B is commute i.e. AB == BA or not 
    (In special case)
    Since we need only one counter example to disprove commutativity.
    First assume that it is True.
    When non commute pattern found, break or return False.
    (In general case)
    Since uniqueness is degenerated in the general case, we can not reject
    and commutativity until we reach the end of all calculation.
    Thats why the result is only returned at the end of the function.
    
    First convert the all Operator into binary representation,
    then carry out commutator.
    Finally convert back to ordinary XYZ string representation.
    
    Using parity check.
    Using 123cyclic index clock system i.e. L%3+1 == R

    """
    def body(intersect, akey, acof, bcof, bkey, debug=0):
        new_key = akey.copy(); new_key.update(bkey)
        clw = aclw = 0
        for i in intersect: 
            l, r = akey[i], bkey[i]
            if l != r:
                new_key[i] = l ^ r
                if l%3+1 == r: 
                    clw += 1
                else: 
                    aclw += 1
            else:
                new_key[i] = None   
                
        if aclw%2 ^ clw%2:
            new_key = tuple(sorted(
                            ((bit,xyz) for bit,xyz in new_key.items() if xyz), key=itemgetter(0)
                            ))
            if ((aclw%2-clw%2) + (clw+aclw)%4)%4:
                if new_key in new:
                    new[new_key] -= acof*bcof 
                else: new[new_key] = -acof*bcof 
            else: 
                if new_key in new:
                    new[new_key] += acof*bcof 
                else: new[new_key] = acof*bcof 


    if not isinstance(A, QubitOperator) \
    and not isinstance(B, QubitOperator):
        return None
        
    new = dict()
    A, B = getrid_QOconstant(binary_pauli(A)), getrid_QOconstant(binary_pauli(B))
    
    ABoverlap = set( a for a in 
                    set(A.keys()).intersection(B.keys()) 
                    if A[a]==B[a] 
                    )
    AdifferentB, BdifferentA = set(A.keys())-ABoverlap, set(B.keys())-ABoverlap
    
    B_hashs = tuple([dict(k),B[k]] for k in BdifferentA)
    Boverlap = tuple([dict(k),B[k]] for k in ABoverlap)

    for akey in ABoverlap:
        akey, aidx, acof = dict(akey), set(map(itemgetter(0), akey)), A[akey]
        for (bkey,bcof) in B_hashs:
            intersect = aidx.intersection(bkey.keys())
            if intersect:
                body(intersect, akey, acof, bcof, bkey, debug)
            
    for akey in AdifferentB:
        akey, aidx, acof = dict(akey), set(map(itemgetter(0), akey)), A[akey]
        for (bkey,bcof) in B_hashs:
            intersect = aidx.intersection(bkey.keys())
            if intersect:
                body(intersect, akey, acof, bcof, bkey, debug)  
        for (bkey,bcof) in Boverlap:
            intersect = aidx.intersection(bkey.keys())
            if intersect:
                body(intersect, akey, acof, bcof, bkey, debug)
    
    if mode == QubitOperator:
        return binary_pauli({k:v*2j for k,v in new.items() if v})

    elif mode == dict: 
        return binary_pauli({k:v*2j for k,v in new.items() if v}).terms
    
    elif mode == bool:
        for v in set(new.values()):
            if v: return False
    return True   
