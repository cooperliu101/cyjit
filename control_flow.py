import ast
from copy import deepcopy as copy

class TypeVal(object):
    def __init__(self, type, def_offset=None):
        self.type=type
        self.def_offset=def_offset
    def __repr__(self):
        return "%s was define at %s"%(self.type, self.def_offset)
    
class InsertPass(object):
    def __init__(self, node):
        self.add_num=0
        self.current_lineno=None
        self.node=node
        
    def run(self):
        self.insert_pass(self.node.body[0].body)
    
    def insert_pass(self, body):
        stmt=None
        for stmt in body:
            stmt.lineno+=self.add_num
            self.current_lineno=stmt.lineno          
            if isinstance(stmt, (ast.While, ast.For, ast.If)):
                self.insert_pass(stmt.body)
                self.insert_pass(stmt.orelse)
                
        if isinstance(stmt, (ast.For, ast.While)):
            pass_node=ast.Pass(lineno=self.current_lineno+1, col_offset=stmt.col_offset)
            body.append(pass_node)
            self.current_lineno=pass_node.lineno
            self.add_num+=1
    
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
        args = self.body, self.outgoing, self.incoming, self.context
        return "block(body: %s, outgoing: %s, incoming: %s, context=%s)" % args

    
map_offset_node={}
map_offset_body={}    

class myControlFlowAnalysis(object):
    def __init__(self, node):
        self.node=node
        
        first_block=CFBlock(offset=-1)
        first_block.body.append(-1)
        self.blocks={-1:first_block}
        
        self._incoming_blocks=[first_block] #for next block
        self._curblock=None
        self._force_new_block=True
        self._loop_next_blocks=[]


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
            

            map_offset_node[offset]=stmt
            map_offset_body[offset]=body
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
        if_outgoing_block=self._visit_body(node.body)

        if node.orelse:
            self._incoming_blocks=entry_blocks
            orelse_outgoing_block=self._visit_body(node.orelse)
            outgoing_blocks=if_outgoing_block+orelse_outgoing_block
        else:
            outgoing_blocks=entry_blocks+if_outgoing_block

        self._force_new_block=True
        return outgoing_blocks
    
    def visit_While(self, node):        
        entry_blocks=[self._curblock]
        while_outgoing_blocks=self._visit_body(node.body)
        for block in while_outgoing_blocks:
            entry_blocks[0].incoming.append(block.offset)
        if node.orelse:
            self._incoming_blocks=entry_blocks
            orelse_outgoing_block=self._visit_body(node.orelse)   
            outgoing_blocks=orelse_outgoing_block
        else:
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


class clearUnreachedNode():
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
            if entry_offset==-1:
                continue            
            block=self.cf.blocks[entry_offset]
            for offset in block.body:
                node=map_offset_node[offset]
                node.keep=True
                
    def run(self):
        funcdef_node=self.cf.node.body[0]
        self.mark()
        for node in funcdef_node.body[:]:
            self.visit_and_clear(node, funcdef_node.body)
import numpy

a=1    
s='''
def f():
  if 1:
    a=0.0
  a
  if 1:
    a=0
  return a
'''
#s='''
#def f():
  #while a:
    #a=1.0
  #b=a
#'''
#s='''
#def f():
  #a=a[0]+1
  #if a>1:
    #a=1.0
  #a=a
#'''
#s='''#1
#def f(): #2
  #b=a
  
#'''      
#s='''#1
#def f(): #2
  #while 1:  #3
    #while 1: #4
      #pass #5
    #if 1: #6
      #while 1: #7
        #break #8
        #return 0 #9
    #else: #10
      #while 1: #11
        #pass #12
      #while 1: #13 
        #pass #14
#'''
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
s='''#1
def f(): #2
  a=1 #3
  if a>1: #4
    b=a #5
  else: #6
    if a>1: #7
      a=1.0 #8
    else: #9
      if a>1: #10
        b=a #11
  b=a #12
'''
node=ast.parse(s)
InsertPass(node).run()
print ast.dump(node, include_attributes=True)

cf=myControlFlowAnalysis(node)
cf.run()
clearUnreachedNode(cf).run()

#print cf.offset_node_map
#print cf.offset_body_map

from cffi import FFI
ffi=FFI()

from numba_types import Array, int32, float64, pyobject, Tuple, is_int_tuple, none, string
import numpy_support

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

map_name_types={}
#def get_order(name, type):
    #if name in map_name_types:
        #types=map_name_types[name]
        #if type in types:
            #return types.index(type)
        #else:
            #types.append(type)
            #return len(types)-1
    #else:
        #map_name_types[name]=[type]
        #return 0   
    
def get_order(name, type):
    if name in map_name_types:
        types, next_order=types_info=map_name_types[name]
        if type in types:
            return types[type]
        else:
            types[type]=next_order
            types_info[1]+=1
            return next_order
    else:
        map_name_types[name]=[{type:0},1]
        return 0
            
 
direct_flag={}    
global_names=set()
load_coerce_infos=set()
class TypeInfer(object):
    def __init__(self, cf, globals):
        self.cf=cf
        self.current_block=None
        self.globals=globals
        self.need_reinfer=False
        
    def run(self):
        need_reinfered_contexts=[]
        self.need_reinfer=False
        for entry_offset in sorted(self.cf.blocks.keys()):
            if entry_offset==-1:
                continue
            self.current_block=current_block=self.cf.blocks[entry_offset]
            incoming_blocks=[]
            for offset in current_block.incoming:
                bk=self.cf.blocks[offset]
                if bk.context=={} and offset>entry_offset:
                    need_reinfered_contexts.append(bk.context)
                    self.need_reinfer=True
                else:
                    incoming_blocks.append(bk)
            self.join(current_block, incoming_blocks)
            context=current_block.context
            for offset in sorted(current_block.body):
                node=map_offset_node[offset]
                fname='visit_'+type(node).__name__
                func=getattr(self, fname, None)
                if func is not None:
                    func(node, context)
                else:
                    print 'no %s'%fname
        
        #check reinfer
        if self.need_reinfer:   
            for context in need_reinfered_contexts:
                if context=={}:
                    self.need_reinfer=False
                    break
                 
    def join(self, current_block, incoming_blocks):
        if len(incoming_blocks)==1:
            current_block.context.update(incoming_blocks[0].context)
            current_block.coerce_names.update(incoming_blocks[0].coerce_names)
            return
        #len(incoming_blocks)>1
        collect={}
        all_names=set([])
        for block in incoming_blocks:
            all_names|=set(block.context.keys())
            
        for name in all_names:
            typevals=[]
            for block in incoming_blocks:
                if name in block.context:
                    typeval=block.context[name]
                elif name in self.globals:
                    value=self.globals[name]
                    type=self.get_value_type(value)
                    typeval=TypeVal(type,
                                    def_offset={-1:type},
                                    )
                    block.context[name]=typeval
                    global_names.add((name, type))
                else:
                    typeval=None
                typevals.append(typeval)
            collect[name]=typevals

        
        for name, typevals in collect.iteritems():
            types=set([])
            def_offset={}
            for typeval in typevals:
                if typeval is None:
                    #warning if load this variable
                    continue
                types.add(typeval.type)
                def_offset.update(typeval.def_offset)
  
            coerce_type=reduce(self.spanning_types, types)
            coerce_typeval=TypeVal(coerce_type, def_offset)
            current_block.context[name]=coerce_typeval

    
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
            
    def get_value_type(self, value):
        if isinstance(value, int):
            return int32
        elif isinstance(value, float):
            return float64
        elif numpy_support.is_array(value):
            dtype = numpy_support.from_dtype(value.dtype)
                        # force C contiguous
            ty = Array(dtype, value.ndim, 'C')        
            return ty
        else:
            return pyobject
        
        
        
    def typeof_Str(self, node, context):
        return string
        
        
    def typeof_Name(self, node, context):
        is_coerce=False
        name=node.id
        if name in context:
            typeval=context[name]
            type=typeval.type
            node.typeval=typeval
            for offset,ty in typeval.def_offset.iteritems(): #change attribuate name
                if ty != type:
                    load_coerce_infos.add((name, type, offset, ty))
                    
            #self.current_block.load_names[name]=typeval
            return type
        ##lookup the name in globals
        ##now only consider int float array
        elif name in self.globals:
            value=self.globals[name]
            type=self.get_value_type(value)
            typeval=TypeVal(type, 
                            def_offset={-1:type},
                            )
            context[name]=typeval
            node.typeval=typeval
            self.current_block.load_names[name]=typeval
            global_names.add((name, type))
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
        for elt in node.elts:
            self.typeof(elt, context)
        return pyobject
    
    def typeof_Tuple(self, node, context):
        items=[]
        for elt in node.elts:
            items.append(self.typeof(elt, context))
        return Tuple(tuple(items))
    
    def typeof_Dict(self, node, context):
        return pyobject
    
    def typeof_Call(self, node, context):
        if isinstance(node.func, ast.Name):
            func_name=node.func.id
            if func_name in ['ones', 'zeros', 'empty']:
                shape=node.args[0]
                #dtype=node.args[1]
                shape_type=self.typeof(shape, context)
                if shape_type==int32:
                    return Array(float64, 1, 'C')
                elif is_int_tuple(shape_type):
                    return Array(float64, shape_type.count, 'C')
                
            elif func_name in ['int', 'float']:
                for arg in node.args:
                    self.typeof(arg, context)
                return {'int':int32,'float':float64}[func_name]
            
            elif func_name in ['new']:
                cdecl=node.args[0]
                assert isinstance(cdecl, ast.Str)
                return ffi._typeof(cdecl.s)
            
        elif isinstance(node.func, ast.Attribute):
            #fix me
            self.typeof(node.func.value, context)
        
 
        for arg in node.args:
            self.typeof(arg, context)            
        
        return pyobject
    
    def typeof_Subscript(self, node, context):
        direct_flag[node]=False
        
        value_type=self.typeof(node.value, context)
        slice_value_type=self.typeof(node.slice.value, context)

        if isinstance(value_type, Array):

            if slice_value_type==int32:
                ndim=value_type.ndim-1
            elif is_int_tuple(slice_value_type):
                ndim=value_type.ndim-slice_value_type.count
            else:
                return pyobject
            
            if ndim==0:
                direct_flag[node]=True
                return value_type.dtype
            return Array(value_type.dtype, ndim, 'C')   
         
        elif isinstance(value_type, ffi.CType):
            assert value_type.kind in ['pointer', 'array']
            if slice_value_type==int32:
                return value_type.item
            
        return pyobject
    
    def typeof_Compare(self, node, context):
        self.typeof(node.left, context)
        self.typeof(node.comparators[0], context)
        
    def visit_If(self, node, context):
        test=node.test
        self.typeof(test, context)

        
        
    def visit_While(self, node, context):
        test=node.test
        self.typeof(test, context)

        
            
    def visit_Expr(self, node, context):
        self.typeof(node, context)
        
    def visit_Assign(self, node, context):
        target=node.targets[0]
        
        value=node.value
        #value_type='object'
        value_type=self.typeof(value, context)
        if isinstance(target, ast.Name):
            target_name=target.id
            target.typeval=context[target_name]=TypeVal(value_type,
                                                        def_offset={node.lineno:value_type})

            
    def visit_AugAssign(self, node, context):
        target=node.target
        value=node.value
        target_type=self.typeof(target, context)
        value_type=self.typeof(value, context)
        coerce_type=self.spanning_types(target_type, value_type)
        target_name=target.id
        target.typeval=context[target_name]=TypeVal(coerce_type, 
                                                    def_offset={node.lineno:coerce_type})

        
    def visit_For(self, node, context):
        for_iter=node.iter
        for_target=node.target
        if isinstance(for_iter, ast.Call):
            if for_iter.func.id in ['range', 'xrange']:
                for_target.typeval=context[for_target.id]=TypeVal(int32,
                                                                  def_offset={node.lineno:int32})
                for arg in for_iter.args:
                    assert self.typeof(arg, context)==int32  
                    
    def visit_Return(self, node, context):
        value=node.value
        self.typeof(value, context)
        
        
#class InsertCoerceNode(object):
    #def __init__(self, cf):
        #self.cf=cf

    #def run(self):
        #for entry_offset in sorted(self.cf.blocks.keys()):
            #if entry_offset==-1:
                #continue
            #current_block=self.cf.blocks[entry_offset]
            #for load_name, load_typeval in current_block.load_names.iteritems():
                #if load_name in current_block.coerce_names:
                    #coerce_typeval=current_block.coerce_names[load_name]
                    #for incoming_block in self._get_incoming_blocks(current_block):
                        #incoming_typeval=incoming_block.context[load_name]
                        #if incoming_typeval.type!=coerce_typeval.type:
                            #insert_node=ast.Assign(targets=[ast.Name(id=load_name, 
                                                                     #ctx=ast.Store, 
                                                                     #typeval=coerce_typeval)],
                                                   #value=ast.Name(id=load_name, 
                                                                  #ctx=ast.Load, 
                                                                  #typeval=incoming_typeval))
                            #self.insert_at_end(incoming_block, insert_node)
                        
    #def _get_incoming_blocks(self, block):
        #return [self.cf.blocks[offset] for offset in block.incoming]    

    #def insert_at_end(self, block, insert_node):
        #end_offset=block.body[-1]
        #if end_offset==-1:
            #body=map_offset_body[min(map_offset_body.keys())]
            #body.insert(0, insert_node)
            #return 
        #node=map_offset_node[end_offset]
        #body=map_offset_body[end_offset]
        #idx=body.index(node)
        #if isinstance(node, (ast.If, ast.Break)):
            ##pre_idx=idx-1
            ##if pre_idx>=0:
            ##    if is_same_node(body[pre_idx], insert_node):
            ##        return 
            #body.insert(idx, insert_node)
        #elif isinstance(node, (ast.While, ast.For)):
            #assert len(block.body)==1
            #for incoming_block in self._get_incoming_blocks(block):
                #self.insert_at_end(incoming_block, copy(insert_node))
                    
        #else:
            
            ##if next_idx<len(body):
                ##if is_same_node(body[next_idx], insert_node):
                    ##return 
            ##if block.end_offset==None:
            ##    block.end_offset=block.body[-1]
            #body.insert(idx+1+block.insert_num, insert_node)
            #block.insert_num+=1
 
class InsertCoerceNode(object):
    def __init__(self, cf):
        self.cf=cf

    def run(self):
        for name,coerce_type,offset,incoming_type in load_coerce_infos:    
            insert_node=ast.Assign(targets=[ast.Name(id=name, 
                                                ctx=ast.Store, 
                                                typeval=TypeVal(coerce_type))],
                                   value=ast.Name(id=name, 
                                             ctx=ast.Load, 
                                             typeval=TypeVal(incoming_type)))
            self.insert_below(offset, insert_node)                
                    
    def _get_incoming_blocks(self, block):
        return [self.cf.blocks[offset] for offset in block.incoming]    

    def insert_below(self, offset, insert_node):
        if offset==-1:
            body=map_offset_body[min(map_offset_body.keys())]
            body.insert(0, insert_node)
            return 
        node=map_offset_node[offset]
        body=map_offset_body[offset]
        idx=body.index(node)
        assert isinstance(node, (ast.Assign, ast.AugAssign))
        body.insert(idx+1, insert_node)
        
        
used_types=set([])
class myRewriter(ast.NodeTransformer): 
    def visit_Name(self, node):
        typeval=getattr(node, 'typeval', None)
        if typeval is not None:
            type=typeval.type
            name=node.id

            if isinstance(type, ffi.CType):
                if ty.kind=='pointer':
                    node.id="%s_%s"%(node.id, type.cname.replace(' ','').replace('*','_p'))
                elif ty.kind=='array':
                    node.id="%s_%s"%(node.id, type.cname.replace('[','_').replace(']',''))
                else:
                    node.id="%s_%s"%(node.id, type.cname)
            else:
                order=get_order(name, type)
                node.id="%s_%s"%(name, order)
                used_types.add((name, type, order))
            
        return node

    def visit_Subscript(self, node):
        value=node.value
        slice_value=node.slice.value        
        self.visit(value)
        self.visit(slice_value)
        if direct_flag[node]:      
            value_type=value.typeval.type
            assert isinstance(value_type, Array)
            assert value_type.ndim>=1
            if isinstance(slice_value, ast.Num):
                assert value_type.ndim==1
                new_node=ast.Name(id="(<%s *>(%s_data_ptr+%s_stride%s*%s))[0]"%(value_type.dtype, value.id, value.id, 0, slice_value.n))
                return new_node
            elif isinstance(slice_value, ast.Name):
                slice_value_type=slice_value.typeval.type
                if slice_value_type == int32:
                    new_node=ast.Name(id="(<%s *>(%s_data_ptr+%s_stride%s*%s))[0]"%(value_type.dtype, value.id, value.id, 0, slice_value.id))
                    return new_node
                elif is_int_tuple(slice_value_type):
                    s='(<%s *>(%s_data_ptr'%(value_type.dtype, value.id)
                    for idx in range(slice_value_type.count):
                        s+='+%s_stride%s*<int>(%s[%s])'%(value.id, idx, slice_value.id, idx)
                    s+='))[0]'
                    new_node=ast.Name(id=s)
                    return new_node
            elif isinstance(slice_value, ast.Tuple):
                assert len(slice_value.elts)==value_type.ndim
                s='(<%s *>(%s_data_ptr'%(value_type.dtype, value.id)
                for idx, elt in enumerate(slice_value.elts):
                    if isinstance(elt, ast.Name):
                        assert elt.type_info[0]==int32
                        s+='+%s_stride%s*%s'%(value.id, idx, elt.id)
                    elif isinstance(elt, ast.Num):
                        assert isinstance(elt.n, int)
                        s+='+%s_stride%s*%s'%(value.id, idx, elt.n)
                    else:
                        raise Exception("%s"%elt)
                s+='))[0]'
                new_node=ast.Name(id=s)
                return new_node
        return node    

    
class InsertArrayInfo(object):
    def __init__(self):
        pass
    
    def insert_below(self, offset, array_name):
        if offset==-1:
            body=map_offset_body[min(map_offset_body.keys())]
            idx_def=-1
        else:
            node=map_offset_node[offset]
            body=map_offset_body[offset]
            idx_def=body.index(node)
            #assert isinstance(node, ast.Assign)
            #target=node.targets[0]
            #typeval=target.typeval
            #assert isinstance(typeval.type, Array)
        data_ptr_node=ast.Expr(value=ast.Name(id="%s_data_ptr = PyArray_BYTES(%s)"%(array_name.id, array_name.id)))
        body.insert(idx_def+1, data_ptr_node) 
        for idx in range(array_name.typeval.type.ndim):
            stride_node=ast.Expr(value=ast.Name(id="%s_stride%s = PyArray_STRIDES(%s)[%s]"%(array_name.id, idx, array_name.id, idx)))
            body.insert(idx_def+idx+2, stride_node)
            
    def run(self):
        for node, flag in direct_flag.iteritems():
            if flag==True:
                assert isinstance(node.value, ast.Name)
                for offset in node.value.typeval.def_offset:
                    self.insert_below(offset, node.value)

class InsertDefination(object):
    def __init__(self, node):
        self.node=node
    def run(self):
        func_body=self.node.body[0].body
        for name, type, order in used_types:
            if isinstance(type, Array):
                for idx in range(type.ndim)[::-1]:
                    defination_node=ast.Expr(value=ast.Name(id="cdef %s %s"%("int", "%s_stride%s"%('%s_%s'%(name,order), idx))))
                    func_body.insert(0, defination_node)                    
                defination_node=ast.Expr(value=ast.Name(id="cdef %s %s"%("char *", "%s_data_ptr"%('%s_%s'%(name,order)))))
                func_body.insert(0, defination_node)     
                if (name, type) in global_names:
                    defination_node=ast.Expr(value=ast.Name(id="cdef %s %s=GLOBALS['%s']"%(pyobject, '%s_%s'%(name,order), name)))
                else:
                    defination_node=ast.Expr(value=ast.Name(id="cdef %s %s"%(pyobject, '%s_%s'%(name,order))))
                func_body.insert(0, defination_node)                
            else:
                if (name, type) in global_names:
                    defination_node=ast.Expr(value=ast.Name(id="cdef %s %s=GLOBALS['%s']"%(type, '%s_%s'%(name,order), name)))
                else:
                    defination_node=ast.Expr(value=ast.Name(id="cdef %s %s"%(type, '%s_%s'%(name,order))))
                func_body.insert(0, defination_node)
            
infer=TypeInfer(cf, globals())
infer.run()
while infer.need_reinfer:
    infer.run()
    
print cf.blocks
InsertCoerceNode(cf).run()
#while infer.need_reinfer:
#    infer.run()
myRewriter().visit(node)
#RewriteSubscript().visit(node)
InsertArrayInfo().run()
InsertDefination(node).run()
print map_name_types
print ast.dump(node)

from astunparse import Unparser
from cStringIO import StringIO
buf=StringIO()
Unparser(node,buf)
print buf.getvalue()
    
#cf.offset_body_map[3].append(ast.Name(id='a'))
#cf.offset_node_map[3].targets[0].id='aa'
#print ast.dump(node.body[0])