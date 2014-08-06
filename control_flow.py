import ast
from copy import deepcopy as copy

class TypeInfo(object):
    def __init__(self, type, def_offsets=None):
        self.type=type
        self.def_offsets=def_offsets
    def __repr__(self):
        return "%s was define at %s"%(self.type, self.def_offsets)
    
class InsertPass(object):
    def __init__(self, env):
        self.add_num=0
        self.current_lineno=None
        self.node=env.node
        
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
            insert_node=ast.Pass(lineno=self.current_lineno+1, col_offset=stmt.col_offset)
            body.append(insert_node)
            self.current_lineno=insert_node.lineno
            self.add_num+=1
    
class InsertReturn(object):
    def __init__(self, env):
        self.node=env.node
        
    def run(self):
        self.insert_return(self.node.body[0].body)
    
    def insert_return(self, body):
        stmt=body[-1]
        if isinstance(stmt, ast.Pass):
            return_none_node=ast.Return(value=ast.Name(id='None'), lineno=stmt.lineno)
            body[-1]=return_none_node
        elif not isinstance(stmt, ast.Return):
            assert not isinstance(stmt, (ast.For, ast.While))
            return_none_node=ast.Return(value=ast.Name(id='None'), lineno=stmt.lineno+1)
            body.append(return_none_node)
        
            
class CFBlock(object):
    def __init__(self, offset):
        self.offset = offset
        self.body = []
        self.outgoing = []
        self.incoming = []
        self.terminating = False
        self.context={}
        #self.insert_num=0

    def __repr__(self):
        args = self.body, self.outgoing, self.incoming, self.context
        return "block(body: %s, outgoing: %s, incoming: %s, context=%s)" % args

class Env(object):
    #__slots__=['']
    def __init__(self):
        self.node=None
        self.map_offset_node={}
        self.map_offset_body={}
        self.arg_names=set()
        self.global_names=set()
        self.return_node_infos=set()
        self.load_coerce_infos=set()
        self.index_rewrite_infos=set()
        self.used_types=set()
           

class myControlFlowAnalysis(object):
    def __init__(self, env):
        env.cf=self
        self.env=env
        self.node=env.node
        self.map_offset_body=env.map_offset_body
        self.map_offset_node=env.map_offset_node
        
        self.blocks={}
        self._incoming_blocks=[] #for next block
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
            

            self.map_offset_node[offset]=stmt
            self.map_offset_body[offset]=body
            yield stmt        

    def run(self):
        funcdef_node=self.node.body[0]
        self.visit_FunctionDef(funcdef_node)
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
    
    def visit_FunctionDef(self, node):
        if self._use_new_block(node):
            self._start_new_block(node)
        offset=node.lineno
        self._curblock.body.append(offset)
        self.map_offset_node[offset]=node
        self.map_offset_body[offset]=None   
        self.map_offset_body[-1]=node.body
        self._visit_body(node.body)
    
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
    def __init__(self, env):
        self.env=env
        self.cf=env.cf
        self.map_offset_node=env.map_offset_node


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
                node=self.map_offset_node[offset]
                node.keep=True
                
    def run(self):
        funcdef_node=self.cf.node.body[0]
        self.mark()
        for node in funcdef_node.body[:]:
            self.visit_and_clear(node, funcdef_node.body)

#a=1    
#s='''
#def f():
  #if 1:
    #a=0.0
  #a
  #if 1:
    #a=0
  #return a
#'''
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


#print cf.offset_node_map
#print cf.offset_body_map

import numpy
from cffi import FFI
ffi=FFI()

from numba_types import Array, int32, float64, pyobject, Tuple, is_int_tuple, none, string, i1, integer_domain, real_domain
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
            
 
#aggresive 
def spanning_types(ty1, ty2):
    if ty1 == ty2:
        return ty1
    if ty1 in real_domain and ty2 in integer_domain: #need rank?
        return ty1
    if ty1 in integer_domain and ty2 in real_domain:
        return ty2

    return pyobject

#no check for broadcast ability 
def spanning_broadcast_types(ty1, ty2):
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
        
def get_value_type(value):
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



class TypeInfer(object):
    def __init__(self, env, args, func_globals):
        self.cf=env.cf
        self.global_names=env.global_names #share
        self.arg_names=env.arg_names #share
        self.map_offset_node=env.map_offset_node
        self.return_node_infos=env.return_node_infos
        self.load_coerce_infos=env.load_coerce_infos
        self.current_block=None
        self.globals=func_globals
        self.args=args
        self.need_reinfer=False
        
    def run(self):
        self.infer()
        while self.need_reinfer:
            self.infer()
    
    def infer(self):
        need_reinfered_contexts=[]
        self.need_reinfer=False
        
        for entry_offset in sorted(self.cf.blocks.keys()):
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
                node=self.map_offset_node[offset]
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
        if not incoming_blocks:
            return
        if len(incoming_blocks)==1:
            current_block.context.update(incoming_blocks[0].context)
            return
        #len(incoming_blocks)>1
        collect={}
        all_names=set([])
        for block in incoming_blocks:
            all_names|=set(block.context.keys())
            
        for name in all_names:
            typeinfos=[]
            for block in incoming_blocks:
                if name in block.context:
                    typeinfo=block.context[name]                  
                elif name in self.globals:
                    value=self.globals[name]
                    type=self.get_value_type(value)
                    typeinfo=TypeInfo(type,
                                    def_offsets={-2:type}, #offset -2 means defination location is globals
                                    )
                    block.context[name]=typeinfo
                    self.global_names.add((name, type))
                else:
                    typeinfo=None
                typeinfos.append(typeinfo)
            collect[name]=typeinfos

        
        for name, typeinfos in collect.iteritems():
            types=set([])
            def_offsets={}
            for typeinfo in typeinfos:
                if typeinfo is None:
                    #warning if load this variable
                    continue
                types.add(typeinfo.type)
                def_offsets.update(typeinfo.def_offsets)
  
            coerce_type=reduce(spanning_types, types)
            coerce_typeinfo=TypeInfo(coerce_type, def_offsets)
            current_block.context[name]=coerce_typeinfo

    
    def typeof(self, node, context):
        node_type=type(node).__name__
        #print node_type
        fn=getattr(self, 'typeof_%s'%node_type, None)
        if fn==None:
            print "cant find typeof_%s"%node_type
        else:
            return fn(node, context)    
        
        
    def typeof_Str(self, node, context):
        return string
        
        
    def typeof_Name(self, node, context):
        name=node.id
        if name=='None':
            node.typeinfo=TypeInfo(type=pyobject)
            return pyobject
        
        
        if name in context:
            typeinfo=context[name]
            type=typeinfo.type
            node.typeinfo=typeinfo
            for offset,ty in typeinfo.def_offsets.iteritems(): #change attribuate name
                #if type==pyobject and isinstance(ty, Array):
                #    continue
                if ty != type:
                    self.load_coerce_infos.add((name, type, offset, ty))
            return type

        elif name in self.globals:
            value=self.globals[name]
            type=self.get_value_type(value)
            typeinfo=TypeInfo(type, 
                            def_offsets={-2:type}, #offset -2 means defination location is globals
                            )
            context[name]=typeinfo
            node.typeinfo=typeinfo
            self.global_names.add((name, type))
            return type
        else:
            print("load name %s is not exist!"%name)
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
            return spanning_broadcast_types(left_type, right_type)
        
        return spanning_types(left_type, right_type)
    
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
    
    #fix me
    def typeof_Attribute(self, node, context):
        self.typeof(node.value, context)
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
                
            elif func_name in ['int', 'float', 'i1']: #fix me! use globals info to decide identifer
                for arg in node.args:
                    self.typeof(arg, context)
                return {'int':int32,'float':float64, 'i1':i1}[func_name]
            
            elif func_name in ['new']:
                cdecl=node.args[0]
                assert isinstance(cdecl, ast.Str)
                return ffi._typeof(cdecl.s)
            
        elif isinstance(node.func, ast.Attribute):
            #fix me
            self.typeof(node.func, context)
            
 
        for arg in node.args:
            self.typeof(arg, context)            
        
        return pyobject
    
    def typeof_Subscript(self, node, context):
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
                node.is_index_rewrite=True
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
    
    def visit_FunctionDef(self, node, context):
        #for
        arguments=node.args
        for arg in arguments.args:
            assert isinstance(arg, ast.Name)
            name=arg.id
            type=self.args[name]
            typeinfo=TypeInfo(type,
                            def_offsets={-1:type}, #offset -1 means defination location is function args
                            )
            arg.typeinfo=typeinfo
            context[name]=typeinfo       
            self.arg_names.add((name, type))
        
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
            target.typeinfo=context[target_name]=TypeInfo(value_type,
                                                        def_offsets={node.lineno:value_type})

            
    def visit_AugAssign(self, node, context):
        target=node.target
        value=node.value
        target_type=self.typeof(target, context)
        value_type=self.typeof(value, context)
        coerce_type=spanning_types(target_type, value_type)
        target_name=target.id
        target.typeinfo=context[target_name]=TypeInfo(coerce_type, 
                                                    def_offsets={node.lineno:coerce_type})

        
    def visit_For(self, node, context):
        for_iter=node.iter
        for_target=node.target
        if isinstance(for_iter, ast.Call):
            if for_iter.func.id in ['range', 'xrange']:
                for_target.typeinfo=context[for_target.id]=TypeInfo(int32,
                                                                  def_offsets={node.lineno:int32})
                for arg in for_iter.args:
                    self.typeof(arg, context)#==int32  
                    
    def visit_Return(self, node, context):
        value=node.value
        type=self.typeof(value, context)
        self.return_node_infos.add((node,type))
        
def insert_below(offset, insert_nodes, env):
    map_offset_node=env.map_offset_node
    map_offset_body=env.map_offset_body
    if offset in [-1, -2]:
        body=map_offset_body[-1] #func_body
        for insert_node in insert_nodes[::-1]:
            body.insert(0, insert_node)
    else:
        node=map_offset_node[offset]
        assert isinstance(node, (ast.Assign, ast.AugAssign))
        body=map_offset_body[offset]
        idx=body.index(node)
        for insert_node in insert_nodes[::-1]:
            body.insert(idx+1, insert_node)            

     

class InsertCoerceNode(object):
    def __init__(self, env):
        self.env=env
        self.load_coerce_infos=env.load_coerce_infos
    def run(self):
        for name,type,offset,ty in self.load_coerce_infos:    
            #type<-ty below offset
            insert_node=ast.Assign(targets=[ast.Name(id=name, 
                                                ctx=ast.Store(), 
                                                typeinfo=TypeInfo(type))],
                                   value=ast.Name(id=name, 
                                             ctx=ast.Load(), 
                                             typeinfo=TypeInfo(ty)))
            insert_below(offset, [insert_node], self.env)      
        
def get_return_type(return_node_infos):
    types=[type for _,type in return_node_infos]
    ret_type=reduce(spanning_types, types)  
    return ret_type

class CoerceReturn(object):
    def __init__(self, env):
        self.env=env
        self.return_node_infos=env.return_node_infos
        
    def run(self):
        #coerce return
        assert len(self.return_node_infos)>0
        if len(self.return_node_infos)==1:
            return 
        #ret_type=get_return_type(self.return_node_infos)
        #for return_node, type in self.return_node_infos:
            #if type!=ret_type:
                #return_value=return_node.value
                #print return_node,type,ret_type
                #return_variable=ast.Name(id='_return_val',
                                         #ctx=ast.Store(),
                                         #typeinfo=TypeInfo(ret_type))
                #coerce_return_nodes=[ast.Assign(targets=[return_variable],
                                                #value=return_value),
                                     #ast.Return(value=copy(return_variable))]#need copy
                #offset=return_node.lineno
                #body=map_offset_body[offset]
                #assert body[-1]==return_node
                #body.pop()    
                #body.extend(coerce_return_nodes)   
                
        

class NameRewrite(ast.NodeVisitor):
    def __init__(self, env):
        self.node=env.node
        self.used_types=env.used_types
        
    def run(self):
        self.visit(self.node.body[0])
        
    def visit_Name(self, node):
        typeinfo=getattr(node, 'typeinfo', None)
        if typeinfo is not None:
            type=typeinfo.type
            #type=pyobject if isinstance(type, Array) else type #array defined as object
            name=node.id

            if isinstance(type, ffi.CType):
                if ty.kind=='pointer':
                    node.id="%s_%s"%(node.id, type.cname.replace(' ','').replace('*','_p'))
                elif ty.kind=='array':
                    node.id="%s_%s"%(node.id, type.cname.replace('[','_').replace(']',''))
                else:
                    node.id="%s_%s"%(node.id, type.cname)
            else:
                if name=='None' and type==pyobject:
                    pass
                else:
                    order=get_order(name, type)
                    node.id="%s_%s"%(name, order)
                    self.used_types.add((name, type, order)) #original type
            
class SubscriptRewrite(ast.NodeTransformer):
    def __init__(self, env):
        self.node=env.node
        self.index_rewrite_infos=env.index_rewrite_infos
        self.env=env
        
    def run(self):
        self.visit(self.node.body[0])
        
    def visit_Subscript(self, node):
        is_index_rewrite=getattr(node,'is_index_rewrite',None)
        if is_index_rewrite:     
            value=node.value
            slice_value=node.slice.value             
            value_type=value.typeinfo.type
            self.index_rewrite_infos.add((value.id,
                                     value.typeinfo.type, 
                                     tuple(value.typeinfo.def_offsets.items()),
                                     ))
            assert isinstance(value_type, Array)
            assert value_type.ndim>=1
            if isinstance(slice_value, ast.Num):
                assert value_type.ndim==1
                new_node=ast.Name(id="(<%s *>(%s_data_ptr+%s_stride%s*%s))[0]"%(value_type.dtype, value.id, value.id, 0, slice_value.n))
                return new_node
            elif isinstance(slice_value, ast.Name):
                slice_value_type=slice_value.typeinfo.type
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
                        assert elt.typeinfo.type==int32
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
    def __init__(self, env):
        self.env=env
        self.index_rewrite_infos=env.index_rewrite_infos
        
    def run(self):
        for array_name, type, def_offsets in self.index_rewrite_infos:
            assert isinstance(type, Array)
            for offset,ty in def_offsets:
                assert ty==type
                insert_nodes=[]
                data_ptr_node=ast.Expr(value=ast.Name(id="cdef char * %s_data_ptr = PyArray_BYTES(%s)"%(array_name, array_name)))
                insert_nodes.append(data_ptr_node)

                for idx in range(type.ndim):
                    stride_node=ast.Expr(value=ast.Name(id="cdef py_ssize_t %s_stride%s = PyArray_STRIDE(%s,%s)"%(array_name, idx, array_name, idx)))    
                    insert_nodes.append(stride_node)
                    
                insert_below(offset, insert_nodes, self.env)

class InsertDefination(object):
    def __init__(self, env):
        self.node=env.node
        self.used_types=env.used_types
        self.global_names=env.global_names
        self.arg_names=env.arg_names
        
        self.env=env
    def run(self):
        functionDef_node=self.node.body[0]
        for arg in functionDef_node.args.args:
            type=arg.typeinfo.type
            if isinstance(type, Array):
                type=pyobject
            arg.id='%s %s'%(type, arg.id)
            
        defination_nodes=[]
        for name, type, order in sorted(self.used_types, key=lambda item:item[2]):
            if (name, type) in self.global_names:
                if isinstance(type, Array):
                    type=pyobject
                global_def_node=ast.Expr(value=ast.Name(id="cdef %s %s=GLOBALS['%s']"%(type, '%s_%s'%(name,order), name)))
                defination_nodes.append(global_def_node)                    
            elif (name, type) in self.arg_names:
                pass #already defined in function head
            else:
                if isinstance(type, Array):
                    type=pyobject                    
                local_def_node=ast.Expr(value=ast.Name(id="cdef %s %s"%(type, '%s_%s'%(name,order))))
                defination_nodes.append(local_def_node)
                
        self.insert_defination(defination_nodes)
                    
    def insert_defination(self, defination_nodes):
        func_body=self.node.body[0].body
        for node in defination_nodes[::-1]:
            func_body.insert(0, node)
                    
        
from inspect import getsource, getargspec

def get_indent(body):      
    ix=0
    while body[ix] in ' \t':
        ix+=1
    return body[:ix]

def get_args(func):
    return getargspec(func).args

def get_func_source(func):
    #without decorate 
    src_lines=[]
    keep=False
    for line in getsource(func).split('\n'):
        if line.strip().startswith('def') and not keep:
            indent_size=len(get_indent(line))
            keep=True
            
        if keep and line.strip():
            src_lines.append(line[indent_size:].rstrip())
    src='\n'.join(src_lines)
    return src

def cyjit(argtypes=[], restype=None):
    def wrap(func):
        env=Env()
        source=get_func_source(func)
        print source
        argnames=get_args(func)
        print argnames
        args={arg_name:arg_type for arg_name,arg_type in zip(argnames, argtypes)}
        
        node=ast.parse(source)
        env.node=node
        
        InsertPass(env).run()
        InsertReturn(env).run()
        print ast.dump(node)
        myControlFlowAnalysis(env).run()
        clearUnreachedNode(env).run()   
        
        TypeInfer(env, args, func.func_globals).run()
        print env.cf.blocks
        InsertCoerceNode(env).run()
        CoerceReturn(env).run()
        print ast.dump(node)
        NameRewrite(env).run()
        SubscriptRewrite(env).run()
        InsertArrayInfo(env).run()
        InsertDefination(env).run()
        print map_name_types

        
        from astunparse import Unparser
        from cStringIO import StringIO
        buf=StringIO()
        Unparser(node,buf)
        print buf.getvalue()      
        print get_return_type(env.return_node_infos)
        
    return wrap            

if __name__ == '__main__':
    #fix me handle restype!!
    a=1
    @cyjit(argtypes=[Array(int32,2,'C')],
           restype=None,) #wather to support locals={'a':int32} like type defination? For now, No
    def f(b):
        s=0
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                s+=b[i,j]
                return s+1
        return s
        