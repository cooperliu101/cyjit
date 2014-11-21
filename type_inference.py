import ast
from numba_types import Array, int32, float64, pyobject, Tuple, is_int_tuple, none, string, i1, integer_domain, real_domain
import numpy_support
from copy import deepcopy as copy

from cffi import FFI
ffi=FFI()

    
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

    
class TypeInfo(object):
    def __init__(self, type, def_offsets=None):
        self.type=type
        self.def_offsets=def_offsets
    
    def __repr__(self):
        return "%s %s"%(self.type, self.def_offsets)
    
    def __eq__(self, other):
        return self.type==other.type


    
    
    
class TypeInfer(object):
    def __init__(self, env, args, func_globals):
        self.cf=env.cf
        self.blocks=env.cf.blocks
        self.global_names=env.global_names #share
        self.arg_names=env.arg_names #share
        self.map_offset_node=env.map_offset_node
        self.return_node_infos=env.return_node_infos
        self.load_coerce_infos=env.load_coerce_infos
        self.current_block=None
        self.globals=func_globals
        self.args=args
        self.need_reinfer=True
        
        self.all_context={}
        
    def run(self):
        self.infer()
        while self.need_reinfer:
            self.infer()
        
    def infer_block(self, b):
        block=self.blocks[b]
        block.context=copy(block.incoming_context)
        context=block.context
        
        for offset in sorted(block.body):
            node=self.map_offset_node[offset]
            fname='visit_'+type(node).__name__
            func=getattr(self, fname, None)
            if func is not None:
                func(node, context)
            else:
                print 'no %s'%fname
                
        recorded_context=self.all_context.get(b, {})
        if not self.same_context(context, recorded_context):
            self.need_reinfer=True 
        self.all_context[b]=context
    
    def same_context(self, context1, context2):
        return context1==context2
    
    #b1.context->b2.incoming_context
    def pass_context_into(self, b1, b2):
        bk1_context=self.blocks[b1].context
        bk2_incoming_context=self.blocks[b2].incoming_context
        for name in bk1_context:
            if name in bk2_incoming_context:
                typeinfo1, typeinfo2=bk1_context[name], bk2_incoming_context[name]
                coerce_type=spanning_types(typeinfo1.type, typeinfo2.type)
                def_offsets={}
                def_offsets.update(typeinfo1.def_offsets)
                def_offsets.update(typeinfo2.def_offsets)
                coerce_typeinfo=TypeInfo(coerce_type, def_offsets)
                bk2_incoming_context[name]=coerce_typeinfo
            else:
                bk2_incoming_context[name]=bk1_context[name]
        
    def infer(self):
        for b1,b2 in self.cf.path:
            self.infer_block(b1)
            self.pass_context_into(b1, b2)
        for b in self.cf.terminate_blocks:
            self.infer_block(b)
        
    def run(self):
        while self.need_reinfer:        
            self.need_reinfer=False
            self.infer()
        

    
    #def infer(self):
        #need_reinfered_contexts=[]
        #self.need_reinfer=False
        
        #for entry_offset in sorted(self.cf.blocks.keys()):
            #self.current_block=current_block=self.cf.blocks[entry_offset]
            #incoming_blocks=[]
            #for offset in current_block.incoming:
                #bk=self.cf.blocks[offset]
                #if bk.context=={} and offset>entry_offset:
                    #need_reinfered_contexts.append(bk.context)
                    #self.need_reinfer=True
                #else:
                    #incoming_blocks.append(bk)
            #self.join(current_block, incoming_blocks)
            #context=current_block.context
            #for offset in sorted(current_block.body):
                #node=self.map_offset_node[offset]
                #fname='visit_'+type(node).__name__
                #func=getattr(self, fname, None)
                #if func is not None:
                    #func(node, context)
                #else:
                    #print 'no %s'%fname
        
        ##check reinfer
        #if self.need_reinfer:   
            #for context in need_reinfered_contexts:
                #if context=={}:
                    #self.need_reinfer=False
                    #break
                 
    #def join(self, current_block, incoming_blocks):
        #if not incoming_blocks:
            #return
        #if len(incoming_blocks)==1:
            #current_block.context.update(incoming_blocks[0].context)
            #return
        ##len(incoming_blocks)>1
        #collect={}
        #all_names=set([])
        #for block in incoming_blocks:
            #all_names|=set(block.context.keys())
            
        #for name in all_names:
            #typeinfos=[]
            #for block in incoming_blocks:
                #if name in block.context:
                    #typeinfo=block.context[name]                  
                #elif name in self.globals:
                    #value=self.globals[name]
                    #type=get_value_type(value)
                    #typeinfo=TypeInfo(type,
                                    #def_offsets={-2:type}, #offset -2 means defination location is globals
                                    #)
                    #block.context[name]=typeinfo
                    #self.global_names.add((name, type))
                #else:
                    #typeinfo=None
                #typeinfos.append(typeinfo)
            #collect[name]=typeinfos

        
        #for name, typeinfos in collect.iteritems():
            #types=set([])
            #def_offsets={}
            #for typeinfo in typeinfos:
                #if typeinfo is None:
                    ##warning if load this variable
                    #continue
                #types.add(typeinfo.type)
                #def_offsets.update(typeinfo.def_offsets)
  
            #coerce_type=reduce(spanning_types, types)
            #coerce_typeinfo=TypeInfo(coerce_type, def_offsets)
            #current_block.context[name]=coerce_typeinfo

    
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
        self.typeof(node.value, context)
        
    def visit_Assign(self, node, context):
        target=node.targets[0]
        
        value=node.value
        #value_type='object'
        value_type=self.typeof(value, context)
        if isinstance(target, ast.Name):
            target_name=target.id
            target.typeinfo=context[target_name]=TypeInfo(value_type,
                                                        def_offsets={node.lineno:value_type})
        elif isinstance(target, ast.Subscript):
            target_type=self.typeof(target, context)
            print target_type, value_type

            
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
      
      