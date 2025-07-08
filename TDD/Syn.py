from TDD.TDD import TDD, Ini_TDD,Clear_TDD,set_index_order,get_unique_table_num,set_root_of_unit,get_count,cont,renormalize,Slicing2,Slicing
from TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs,gen_cir
from TDD.TN import Index,Tensor,TensorNetwork
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import networkx as nx
import itertools
import copy

to_test = False

class Gate:
    def __init__(self,name='',q_c={},q_t=0,para=np.eye(2)):
        self.name = name
        self.q_c = q_c
        self.q_t = q_t
        self.nodes = {}
        X = np.array([[0,1],[1,0]])
        if 'x' in name:
            self.para = X
        elif 'p' in name:
            if type(para) == float:
                self.para = np.array([[1,0],[0,np.exp(1j*para)]])
            else:
                self.para = para
        else:
            self.para=para

        if 'u' in name:
            if np.allclose(para,X):
                self.name = 'x'
                self.para = X
            elif abs(para[0][1])<1e-10 and abs(para[1][0])<1e-10:
                self.name = 'p'
        if np.allclose(self.para,np.eye(2)):
            self.name = 'p'
            self.q_c={}
            self.q_t = 0
            
        
    def qubits(self):
        return list(self.q_c.keys())+[self.q_t]
            
    def __eq__(self,other):
        return self.q_c==other.q_c and self.q_t == other.q_t and np.allclose(self.para,other.para)
        
    def __str__(self):
        dict_str = '('+", ".join([f"{key}: {value}" for key, value in self.q_c.items()])+')'
        return str((self.name,dict_str,self.q_t,self.para))
        
    def to_cir(self,n):
        cir = QuantumCircuit(n)
        if 'p' in self.name:
            theta = np.angle(self.para[1][1])
            cir.p(theta,self.q_t)
        elif 'x' in self.name:
            cir.x(self.q_t)
        elif 'u' in self.name:
            u3 = UnitaryGate(self.para)
            cir.append(u3,[self.q_t])
        if len(self.q_c)>0:
            cir_new = QuantumCircuit(n)
            sorted_q_c = dict(sorted(self.q_c.items(), reverse=True))
            ctrl_cond = ''
            new_qubits = []
            for k in sorted_q_c:
                ctrl_cond+=str(sorted_q_c[k]) 
                new_qubits.append(k)
            new_qubits.append(self.q_t)
            # print(ctrl_cond,new_qubits)
            for g, qargs, cargs in cir.data:
                cg = g.control(len(self.q_c),ctrl_state=''.join(reversed(ctrl_cond)))
                cir_new.append(cg,new_qubits)
                return cir_new
        return cir

class Circuit:
    def __init__(self,num_qubits=1,data=[]):
        self.num_qubits = num_qubits
        self.data = data
    def __str__(self):
        s = '['
        for g in self.data:
            s+=str(g)
            s+=','
        s+=']'
        return s      
    def to_cir(self):
        cir = QuantumCircuit(self.num_qubits)
        for g in self.data:
            cir=cir&g.to_cir(self.num_qubits)
        return cir

class CircuitBlock:
    def __init__(self,type= '',num_qubits=1,data=[]):
        self.type = type
        self.num_qubits = num_qubits
        self.data = data
    def __str__(self):
        s = '['
        for g in self.data:
            s+=str(g)
            s+=','
        s+=']'
        return s      
    def to_cir(self):
        cir = QuantumCircuit(self.num_qubits)
        for g in self.data:
            cir=cir&g.to_cir(self.num_qubits)
        return cir

    # def draw(self):
    #     self.to_cir().draw('mpl')

class Node_P:
    def __init__(self,qubit,pos):
        self.qubit = qubit
        self.pos = pos
        self.successor = None
        self.predecessor= None
        self.gates = []
        self.eig_direction = None
    def self_copy(self):
        temp = Node_P(self.qubit,self.pos)
        temp.successor = self.successor
        temp.predecessor = self.predecessor
        temp.gates = copy.copy(self.gates)
        temp.eig_direction = self.eig_direction
        
class H_Graph:
    def __init__(self,num_qubits=1):
        self.num_qubits = num_qubits
        self.start_nodes = {}
        self.end_nodes = {}
        self.nodes = []
        self.gates = []
        
    def to_graph(self):
        temp_node = copy.copy(self.start_nodes)
        G = nx.Graph()
        pos = [0]*self.num_qubits
        is_end = False
        while(not is_end):
            is_end = True
            for k in range(self.num_qubits):
                if temp_node[k]!=None:
                    is_end = False
            for k in range(self.num_qubits):
                if temp_node[k]==None:
                    continue
                G.add_node(temp_node[k],label = (k,pos[k]))
                if temp_node[k]!=self.start_nodes[k]:
                    G.add_edge(temp_node[k],temp_node[k].predecessor)
                    for g in temp_node[k].gates:
                        if len(g.nodes)>1:
                            for (a,b) in itertools.combinations(g.nodes.values(), 2):
                                if a in G.nodes() and b in G.nodes:
                                    G.add_edge(a,b)
                temp_node[k] = temp_node[k].successor
                pos[k]=pos[k]+1
        # G2 = G.to_graph()
        # nx.draw(G2,with_labels = True,labels={node: data['label'] for node, data in G2.nodes(data=True)})                
        return G
        
def get_controlled_circuit2(cir,cond = {0:1},add_qubits_num = 0):
    cir.num_qubits += add_qubits_num
    new_data = []
    for g in cir.data:
        for c in cond:
            g.q_c[c]=cond[c]
    return cir

# g = Gate('x',{},0)
# c = Circuit(2,[g])
# c.to_cir().draw('mpl')
# c2 = get_controlled_circuit2(c,{2:0,1:1},1)
# c.to_cir().draw('mpl')


def simulate(cir,ini=False):
    tn,all_indexs = cir_2_tn(cir)
    tn.tensors=[ts for ts in tn.tensors if ts.name!='nu_q']
#     A=[ts.name for ts in tn.tensors]
#     print(A)
    if ini:
        var=[]
        for idx in all_indexs:
            if idx[0]=='x' and not '_' in idx:
                var.append('a'+idx[1:])
            var.append(idx)
            if idx[0]=='y' and not '_' in idx:
                var.append('b'+idx[1:])
#         print(var)
        for n in range(20):
            s ='a'+str(n)
            if not s in var:
                var.append(s)
            s ='x'+str(n)
            if not s in var:
                var.append(s)            
            for k in range(50):
                s ='x'+str(n)+'_'+str(k)
                if not s in var:
                    var.append(s)
                s ='y'+str(n)+'_'+str(k)
                if not s in var:
                    var.append(s)    
            s ='y'+str(n)
            if not s in var:
                var.append(s)
            s ='b'+str(n)
            if not s in var:
                var.append(s)                        
        var.reverse()
        Ini_TDD(index_order=var)
        set_root_of_unit(2**3)
    tdd=tn.cont()
    return tdd

def change_index(tdd,x,y):
    """change the index from x to y"""
    
    idx_set = []
    key_2_index  = dict()
    index_2_key = dict()
    
    for idx in tdd.index_set:
        if idx.key[0]==x:
            idx_set.append(Index(y+idx.key[1:],idx.idx))
        else:
            idx_set.append(idx)
    
    for k in tdd.key_2_index:
        if isinstance(tdd.key_2_index[k],str) and tdd.key_2_index[k][0]==x:
            key_2_index[k]=y+tdd.key_2_index[k][1:]
        else:
            key_2_index[k]=tdd.key_2_index[k]
            
    for k in tdd.index_2_key:
        if isinstance(k,str) and k[0]==x:
            index_2_key[y+k[1:]]=tdd.index_2_key[k]
        else:
            index_2_key[k]=tdd.index_2_key[k]
    tdd.index_set=idx_set
    tdd.key_2_index=key_2_index
    tdd.index_2_key = index_2_key
    
    
def update_tdd(tdd,cir_head,cir_end):
    tdd1 = simulate(cir_head)
    change_index(tdd1,'x','a')
    change_index(tdd1,'y','x')

    tdd2 = simulate(cir_end)
    change_index(tdd2,'y','b')
    change_index(tdd2,'x','y')
    
    tdd_new=cont(tdd,tdd1)
    tdd_new=cont(tdd_new,tdd2)

    change_index(tdd_new,'a','x')
    change_index(tdd_new,'b','y')
    # tdd_new=renormalize(tdd_new)
    return tdd_new

def get_downward_k_level_map(tdd,k,c):
    if k==0:
        if c==0:
            return tdd.node.out_maps[1],tdd.node.out_weight[1]
        else:
            return tdd.node.successor[0].out_maps[1],tdd.node.successor[0].out_weight[1]
    temp_tdd = Slicing2(Slicing2(tdd,tdd.node.key,0),tdd.node.key-1,0)
    return get_downward_k_level_map(temp_tdd,k-1,c)

def get_branch_dd(u,c):
    bran_tdd = TDD(u.successor[c])
    n2 = bran_tdd.node.key+1
    bran_tdd.index_set = []
    bran_tdd.key_2_index ={-1:-1}
    bran_tdd.key_width = {}
    for k in range(n2):
        bran_tdd.index_set.append(Index('y'+str(k),0))
        bran_tdd.key_2_index[k] = 'y'+str(k)
        bran_tdd.key_width[k] = 2
    bran_tdd.index_2_key = {bran_tdd.key_2_index[a]:a for a in bran_tdd.key_2_index} 
    return bran_tdd


def eliminate_global_map(tdd):
    if to_test:
        print('global')   

    n = tdd.node.key+1
    cir = Circuit(n,[])
    
    the_map=tdd.map

    while the_map.level>-1:
        q = the_map.level
        if the_map.x==1:
            # cir.x(q)
            cir.data.append(Gate('x',{},q))
        if the_map.rotate>0:
            # cir.p(np.pi/4*(8-the_map.rotate),q)
            cir.data.append(Gate('p',{},q,np.pi/4*(8-the_map.rotate)))
        the_map=the_map.father
    return cir

def eliminate_border_control(tdd):
    
    if to_test:
        print('border')
        
    if tdd.node.key==-1:
        return Circuit(0,[])

    n = tdd.node.key+1
    cir = Circuit(n,[])

    u = tdd.node
    the_map = u.out_maps[1]
    
    while the_map.level>-1:
        q1 = u.key
        q = the_map.level
        if the_map.x==1:
            # cir.cx(q1,q)
            cir.data.append(Gate('x',{q1:1},q))
        if the_map.rotate>0:
            # cir.cp(np.pi/4*(8-the_map.rotate),q1,q)
            cir.data.append(Gate('p',{q1:1},q,np.pi/4*(8-the_map.rotate)))
        the_map=the_map.father

    bran_tdd = get_branch_dd(u,0)
    cir_end_t0 = eliminate_border_control(bran_tdd)

    if tdd.node.successor[0]!=tdd.node.successor[1]:
        bran_tdd = get_branch_dd(u,1)
        cir_end_t1 = eliminate_border_control(bran_tdd)
        cir_end_t0 = get_controlled_circuit2(cir_end_t0,{u.key:0},1)
        cir_end_t1 = get_controlled_circuit2(cir_end_t1,{u.key:1},1)
        # cir=cir&cir_end_t0
        # cir=cir&cir_end_t1
        cir.data+=cir_end_t0.data
        cir.data+=cir_end_t1.data
    else:
        cir.data+=cir_end_t0.data
    return cir


def eliminate_border_control_x(tdd):
    
    if to_test:
        print('border')
        
    if tdd.node.key==-1:
        return Circuit(0,[])

    n = tdd.node.key+1
    cir = Circuit(n,[])

    u = tdd.node
    the_map = u.out_maps[1]
    
    while the_map.level>-1:
        q1 = u.key
        q = the_map.level
        if the_map.x==1:
            # cir.cx(q1,q)
            cir.data.append(Gate('x',{q1:1},q))
        # if the_map.rotate>0:
        #     cir.cp(np.pi/4*(8-the_map.rotate),q1,q)
        the_map=the_map.father

    bran_tdd = get_branch_dd(u,0)
    cir_end_t0 = eliminate_border_control_x(bran_tdd)

    if tdd.node.successor[0]!=tdd.node.successor[1]:
        bran_tdd = get_branch_dd(u,1)
        cir_end_t1 = eliminate_border_control_x(bran_tdd)
        cir_end_t0 = get_controlled_circuit2(cir_end_t0,{u.key:0},1)
        cir_end_t1 = get_controlled_circuit2(cir_end_t1,{u.key:1},1)
        cir.data+=cir_end_t0.data
        cir.data+=cir_end_t1.data
    else:
        cir.data+=cir_end_t0.data
    return cir

def eliminate_border_control_p(tdd):
    
    if to_test:
        print('border')
        
    if tdd.node.key==-1:
        return Circuit(0,[])

    n = tdd.node.key+1
    cir = Circuit(n,[])

    u = tdd.node
    the_map = u.out_maps[1]
    
    while the_map.level>-1:
        q1 = u.key
        q = the_map.level
        if the_map.rotate>0:
            # cir.cp(np.pi/4*(8-the_map.rotate),q1,q)
            cir.data.append(Gate('p',{q1:1},q,np.pi/4*(8-the_map.rotate)))
        the_map=the_map.father

    bran_tdd = get_branch_dd(u,0)
    cir_end_t0 = eliminate_border_control_p(bran_tdd)

    if tdd.node.successor[0]!=tdd.node.successor[1]:
        bran_tdd = get_branch_dd(u,1)
        cir_end_t1 = eliminate_border_control_p(bran_tdd)
        cir_end_t0 = get_controlled_circuit2(cir_end_t0,{u.key:0},1)
        cir_end_t1 = get_controlled_circuit2(cir_end_t1,{u.key:1},1)
        cir.data+=cir_end_t0.data
        cir.data+=cir_end_t1.data
    else:
        cir.data+=cir_end_t0.data
    return cir

def eliminate_weight(tdd):
    
    if to_test:
        print('weight')
        
    if tdd.node.key==-1:
        return Circuit(0,[]),1

    n = tdd.node.key+1
    cir = Circuit(n,[])
    u = tdd.node
    bran_tdd = get_branch_dd(u,0)
    cir_end_t0, hold_prob0 = eliminate_weight(bran_tdd)
    if tdd.node.successor[0]!=tdd.node.successor[1]:
        bran_tdd = get_branch_dd(u,1)
        cir_end_t1, hold_prob1 = eliminate_weight(bran_tdd)
        cir_end_t0 = get_controlled_circuit2(cir_end_t0,{u.key:0},1)
        cir_end_t1 = get_controlled_circuit2(cir_end_t1,{u.key:1},1)
        cir.data+=cir_end_t0.data
        cir.data+=cir_end_t1.data
    else:
        cir.data+=cir_end_t0.data
        hold_prob1 = hold_prob0

    w0 = u.out_weight[0]*hold_prob0
    w1 = u.out_weight[1]*hold_prob1
    a=w1/w0
    if a!=0:
        if a==1:
            # cir.h(u.key)
            M = 1/np.sqrt(1+abs(a)**2)*np.array([[1,1],[1,-1]])
        else:
            M = 1/np.sqrt(1+abs(a)**2)*np.array([[1,np.conj(a)],[-a,1]])
        # u3 = UnitaryGate(M)
        # cir.append(u3,[u.key])
        cir.data.append(Gate('u',{},u.key,M))
    hold_prob = w0*np.sqrt(1+abs(a)**2)
    return cir, hold_prob

