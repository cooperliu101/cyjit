import ast
from copy import deepcopy as copy

class CFBlock(object):
    def __init__(self, offset):
        self.offset = offset
        self.body = []
        self.outgoing = []
        self.incoming = []
        self.terminating = False
        self.context={}
        self.coerce_names={}
        self.load_names={}
        self.insert_num=0

    def __repr__(self):
        args = self.body, self.outgoing, self.incoming
        return "block(body: %s, outgoing: %s, incoming: %s)" % args

    def __iter__(self):
        return iter(self.body)

class myControlFlowAnalysis(object):
    def __init__(self, node):
        self.node=node
        self.blocks={}
        self.blockseq=[]
        self._incoming_blocks=[] #for next block
        self._curblock=None
        self._force_new_block=True
        self._loop_next_blocks=[]
        self.offset_node_map={}
        self.offset_body_map={}

    def _use_new_block(self, stmt):
        if type(stmt).__name__ in ['While', 'For']:
            res=True
        else:
            res = self._force_new_block
        self._force_new_block = False
        return res

    def _start_new_block(self, stmt):
        offset=stmt.lineno
        if offset not in self.blocks:
            newblock=CFBlock(offset)
            self.blocks[offset]=newblock
        else:
            newblock=self.blocks[offset]
            
        for block in self._incoming_blocks:
            newblock.incoming.append(block.offset)
            
        self._incoming_blocks=[newblock]
        self._curblock = newblock
        #self.blockseq.append(offset)

           
    def _iter_body(self, body):
        for idx,stmt in enumerate(body):
            
            if type(stmt).__name__ in ['While', 'For']:
                loop_next_stmt=body[idx+1]
                offset=loop_next_stmt.lineno
                loop_next_block=CFBlock(offset)
                self.blocks[offset]=loop_next_block
                self._loop_next_blocks.append(loop_next_block)
            
            if self._use_new_block(stmt):
                self._start_new_block(stmt)
                
            offset=stmt.lineno
            self._curblock.body.append(offset)
            

            self.offset_node_map[offset]=stmt
            self.offset_body_map[offset]=body
            yield stmt        

    def run(self):
        self._visit_body(self.node.body[0].body)
        #fill
        for offset1, block in self.blocks.iteritems():
            for offset2 in block.incoming:
                outgoing=self.blocks[offset2].outgoing
                if offset1 not in outgoing:
                    outgoing.append(offset1)
                    
            for offset2 in block.outgoing:
                incoming=self.blocks[offset2].incoming
                if offset1 not in incoming:
                    incoming.append(offset1)
                    
       
    def _visit_body(self, body):
        self._force_new_block=True
        outgoing_blocks=[]
        for stmt in self._iter_body(body):
            stmt_type=type(stmt).__name__
            fname = "visit_%s" % stmt_type
            fn = getattr(self, fname, None)
            if fn is not None:
                outgoing_blocks=fn(stmt)
            else:
                outgoing_blocks=[self._curblock]         
            self._incoming_blocks=outgoing_blocks
            
            if stmt_type in ['Break', 'Return']:
                return outgoing_blocks   
            
        return outgoing_blocks
                
    def visit_If(self, node):
        entry_blocks=[self._curblock]
        ifblock=self._visit_body(node.body)

        if node.orelse:
            self._incoming_blocks=entry_blocks
            orelseblock=self._visit_body(node.orelse)
            outgoing_blocks=ifblock+orelseblock
        else:
            outgoing_blocks=entry_blocks+ifblock

        self._force_new_block=True
        return outgoing_blocks
    
    def visit_While(self, node):        
        entry_blocks=[self._curblock]
        whileblock=self._visit_body(node.body)
        for block in whileblock:
            entry_blocks[0].incoming.append(block.offset)
        outgoing_blocks=entry_blocks
        self._loop_next_blocks.pop()
        self._force_new_block=True
        return outgoing_blocks
    
    visit_For=visit_While
    
    def jump(self, target):
        self._curblock.outgoing.append(target.offset)
        
    def visit_Break(self, node):
        self.jump(self._loop_next_blocks[-1])
        outgoing_blocks=[]
        return outgoing_blocks
    
    def visit_Return(self, node):
        self._curblock.terminating = True
        outgoing_blocks=[]
        return outgoing_blocks


class clearUnreachNode():
    def __init__(self, cf):
        self.cf=cf

    def visit_and_clear(self, node, body):
        keep=getattr(node, 'keep', None)
        if not keep:
            body.remove(node)
            return             
        body=getattr(node,'body',None)
        if body is not None:
            for node in body[:]:
                self.visit_and_clear(node, body)

    def mark(self):
        for entry_offset in self.cf.blocks.keys():
            block=self.cf.blocks[entry_offset]
            for offset in block.body:
                node=self.cf.offset_node_map[offset]
                node.keep=True
                
    def run(self):
        funcdef_node=self.cf.node.body[0]
        self.mark()
        for node in funcdef_node.body[:]:
            self.visit_and_clear(node, funcdef_node.body)
        
        
s='''#1
def f(): #2
  a=1.0 #3
  while a>1: #4
    a=1 #5
    if a>1: #6
      a=[] #7
      break #8
    a=a+1#9
  if a:
    b=int(a)
'''
#s='''#1
#def f(): #2
  #a=1.0 #3
  #b=1 #4
  #while a<2: #5
    #a=1 #6
    #a=a+1
    #b=a+1.0 #7
    #if a>1: #8
      #a=a+1 #9
      #break #10
  #b=a+b #11
#'''           
    

#s='''#1
#def f(): #2
  #a=1 #3
  #if a>1: #4
    #a=1.0 #5
  #b=a #6
#'''       
      
#s=''' #1
#def f(): #2
  #a=1 #3
  #if a>1: #4
    #a=1.0
    #if a>1: #5
      #b=1.0 #6
  #b=a+b #7
#'''  
#s=''' #1
#def f(): #2
  #a=1 #3
  #if a>1: #4
    #b=1 #5
    #if a>2: #6
      #b=2 #7
    #b=3 #8
  #b=4 #9
#'''
#s='''#1
#def f(): #2
  #a=1 #3
  #if a>1: #4
    #b=a #5
  #else: #6
    #if a>1: #7
      #a=1.0 #8
    #else: #9
      #if a>1: #10
        #b=a #11
  #b=a #12
#'''
node=ast.parse(s)
cf=myControlFlowAnalysis(node)
cf.run()
clearUnreachNode(cf).run()
print cf.blocks
#print cf.offset_node_map
#print cf.offset_body_map



from numba_types import Array, int32, float64, pyobject, Tuple, is_int_tuple, none

def is_same_node(node1, node2):
    if type(node1) != type(node2):
        return False
    assert isinstance(node1, ast.Assign)
    
    target1=node1.targets[0]
    value1=node1.value
    target2=node2.targets[0]
    value2=node2.value   
    if target1.id != target2.id:
        return False
    if target1.type != target2.type:
        return False
    if value1.id != value2.id:
        return False
    if value1.type != value2.type:
        return False
    return True

    
class TypeInfer(object):
    def __init__(self, cf):
        self.cf=cf
        self.current_block=None
        self.direct_flag={}
        
    def run(self):
        #self.need_reinfer=False
        for entry_offset in sorted(self.cf.blocks.keys()):
            self.current_block=current_block=self.cf.blocks[entry_offset]
            incoming_blocks=[]
            for offset in current_block.incoming:
                bk=self.cf.blocks[offset]
                incoming_blocks.append(bk)
                #if bk.context=={}:
                #    self.need_reinfer=True
            self.join(current_block, incoming_blocks)
            context=current_block.context
            for offset in sorted(current_block.body):
                
                node=cf.offset_node_map[offset]
                fname='visit_'+type(node).__name__
                func=getattr(self, fname, None)
                if func is not None:
                    func(node, context)
                else:
                    print 'no %s'%fname

            
    def join(self, current_block, incoming_blocks):
        collect={}
        for block in incoming_blocks:
            for name, type in block.context.iteritems():
                collect.setdefault(name, []).append((type, block))
        
        for name, items in collect.iteritems():
            types=[type for type, _ in items]
            coerce_type=reduce(self.spanning_types, types)
            current_block.context[name]=coerce_type
            for type in types:
                if type != coerce_type:
                    current_block.coerce_names[name]=coerce_type
                    break

    
    def typeof(self, node, context):
        node_type=type(node).__name__
        #print node_type
        fn=getattr(self, 'typeof_%s'%node_type, None)
        if fn==None:
            print "cant find typeof_%s"%node_type
        else:
            return fn(node, context)
        
    #aggresive 
    def spanning_types(self, ty1, ty2):
        if ty1 == ty2:
            return ty1
        if ty1 in [float64, int32] and ty2 in [float64, int32]:
            return float64
        return pyobject
    
    #no check for broadcast ability 
    def spanning_broadcast_types(self, ty1, ty2):
        assert isinstance(ty1, Array) or isinstance(ty2, Array)
        if isinstance(ty1, Array) and isinstance(ty2, Array):
            if ty1.ndim >= ty2.ndim:
                return ty1
            else:
                return ty2
        else:
            if isinstance(ty1, Array):
                if ty2 in [int32, float64]:
                    return ty1
                else:
                    raise Exception("Cannot infer broadcast types (%s,%s)"%(ty1, ty2))
            elif isinstance(ty2, Array):
                if ty1 in [int32, float64]:
                    return ty2
                else:
                    raise Exception("Cannot infer broadcast types (%s,%s)"%(ty1, ty2))
            else:
                raise Exception("should not in")
            
    
                
    def typeof_Name(self, node, context):
        assert isinstance(node.ctx, ast.Load) 
        name=node.id
        if name in context:
            type=context[name]
            node.type=type            
            self.current_block.load_names[name]=type
            return type
        return none

    def typeof_Num(self, node, context):
        if isinstance(node.n, int):
            return int32
        elif isinstance(node.n, float):
            return float64
        
    def typeof_Expr(self, node, context):
        return self.typeof(node.value, context)
    
    def typeof_BinOp(self, node, context):
        left_type=self.typeof(node.left, context)
        right_type=self.typeof(node.right, context)
        if isinstance(left_type, Array) or isinstance(right_type, Array):
            return self.spanning_broadcast_types(left_type, right_type)
        
        return self.spanning_types(left_type, right_type)
    
    def typeof_List(self, node, context):
        return pyobject
    
    def typeof_Tuple(self, node, context):
        items=[]
        for elt in node.elts:
            items.append(self.typeof(elt))
        return Tuple(tuple(items))
    
    def typeof_Dict(self, node, context):
        return pyobject
    
    def typeof_Call(self, node, context):
        func_name=node.func.id
        if func_name in ['ones', 'zeros', 'empty']:
            shape=node.args[0]
            #dtype=node.args[1]
            shape_type=self.typeof(shape)
            if shape_type==int32:
                return Array(float64, 1, 'C')
            elif is_int_tuple(shape_type):
                return Array(float64, shape_type.count, 'C')
            
        elif func_name in ['int', 'float']:
            for arg in node.args:
                self.typeof(arg, context)
            return {'int':int32,'float':float64}[func_name]
        
        else:
            for arg in node.args:
                self.typeof(arg, context)            
        
        return pyobject
    
    def typeof_Subscript(self, node, context):
        self.direct_flag[node]=False
        
        value=node.value
        slice_value=node.slice.value
        if isinstance(value, ast.Name):
            value_type=self.typeof_Name(value)
            if isinstance(value_type, Array):
                slice_value_type=self.typeof(slice_value)
               
                if slice_value_type==int32:
                    ndim=value_type.ndim-1
                elif is_int_tuple(slice_value_type):
                    ndim=value_type.ndim-slice_value_type.count
                else:
                    return pyobject
                
                if ndim==0:
                    self.direct_flag[node]=True
                    return float64
                return Array(float64, ndim, 'C')   
            
        return pyobject
    
    def typeof_Compare(self, node, context):
        self.typeof(node.left, context)
        self.typeof(node.comparators[0], context)
        
    def visit_If(self, node, context):
        test=node.test
        self.typeof(test, context)

        
        
    def visit_While(self, node, context):
        test=node.test
        if isinstance(test, ast.Compare):
            self.typeof(test.left, context)
            self.typeof(test.comparators[0], context)
            
    def visit_Expr(self, node, context):
        self.typeof(node, context)
        
    def visit_Assign(self, node, context):
        target=node.targets[0]
        value=node.value
        #value_type='object'
        value_type=self.typeof(value, context)
        if isinstance(target, ast.Name):
            context[target.id]=value_type
            target.type=value_type
            
    def visit_AugAssign(self, node, context):
        target=node.target
        value=node.value
        target_type=self.typeof(target, context)
        value_type=self.typeof(value, context)
        coerce_type=self.spanning_types(target_type, value_type)
        context[target.id]=coerce_type
        target.type=coerce_type
        
    def visit_For(self, node, context):
        for_iter=node.iter
        for_target=node.target
        if isinstance(for_iter, ast.Call):
            if for_iter.func.id in ['range', 'xrange']:
                context[for_target.id]=int32
                for_target.type=int32
                for arg in for_iter.args:
                    self.typeof(arg, context)    
                    
    def visit_Return(self, node, context):
        value=node.value
        self.typeof(value, context)
    
class InsertCoerceNode(object):
    def __init__(self, cf):
        self.cf=cf
        

    def run(self):
        for entry_offset in sorted(self.cf.blocks.keys()):
            current_block=self.cf.blocks[entry_offset]
            for load_name, load_type in current_block.load_names.iteritems():
                if load_name in current_block.coerce_names:
                    coerce_type=current_block.coerce_names[load_name]
                    assert coerce_type==load_type
                    for incoming_block in self._get_incoming_blocks(current_block):
                        incoming_type=incoming_block.context[load_name]
                        if incoming_type!=coerce_type:
                            insert_node=ast.Assign(targets=[ast.Name(id=load_name, ctx=ast.Store, type=coerce_type)],
                                                   value=ast.Name(id=load_name, ctx=ast.Load, type=incoming_type))
                            self.insert_at_end(incoming_block, insert_node)
                        
    def _get_incoming_blocks(self, block):
        return [self.cf.blocks[offset] for offset in block.incoming]    

    def insert_at_end(self, block, insert_node):
        end_offset=block.body[-1]
        node=self.cf.offset_node_map[end_offset]
        body=self.cf.offset_body_map[end_offset]
        idx=body.index(node)
        if isinstance(node, (ast.If, ast.Break)):
            #pre_idx=idx-1
            #if pre_idx>=0:
            #    if is_same_node(body[pre_idx], insert_node):
            #        return 
            body.insert(idx, insert_node)
        elif isinstance(node, (ast.While, ast.For)):
            assert len(block.body)==1
            for incoming_block in self._get_incoming_blocks(block):
                self.insert_at_end(incoming_block, copy(insert_node))
                    
        else:
            
            #if next_idx<len(body):
                #if is_same_node(body[next_idx], insert_node):
                    #return 
            #if block.end_offset==None:
            #    block.end_offset=block.body[-1]
            body.insert(idx+1+block.insert_num, insert_node)
            block.insert_num+=1
 
type_collect=set([])
class myRewriter(ast.NodeVisitor): 
    def visit_Name(self, node):
        ty=getattr(node, 'type', None)
        if ty is not None:
            type_collect.add((node.id, ty))
            node.id="%s_%s"%(node.id, ty)
            
    
        
        
infer=TypeInfer(cf)
infer.run()
infer.run()

InsertCoerceNode(cf).run()
#while infer.need_reinfer:
#    infer.run()
myRewriter().visit(node)
print type_collect

from astunparse import Unparser
from cStringIO import StringIO
buf=StringIO()
Unparser(node,buf)
print buf.getvalue()
    
#cf.offset_body_map[3].append(ast.Name(id='a'))
#cf.offset_node_map[3].targets[0].id='aa'
#print ast.dump(node.body[0])