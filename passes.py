import ast
from numba_types import Array, int32, float64, pyobject, Tuple, is_int_tuple, none, string, i1, integer_domain, real_domain
from type_inference import TypeInfo
#need to fix
from cffi import FFI
ffi=FFI()

#class InsertPass(object):
    #def __init__(self, env):
        #self.add_num=0
        #self.current_lineno=None
        #self.node=env.node
        
    #def run(self):
        #self.insert_pass(self.node.body[0].body)
    
    #def insert_pass(self, body):
        #stmt=None
        #for stmt in body:
            #stmt.lineno+=self.add_num
            #self.current_lineno=stmt.lineno          
            #if isinstance(stmt, (ast.While, ast.For, ast.If)):
                #self.insert_pass(stmt.body)
                #self.insert_pass(stmt.orelse)
                
        ##check last stmt
        #if isinstance(stmt, (ast.For, ast.While)):
            #insert_node=ast.Pass(lineno=self.current_lineno+1, col_offset=stmt.col_offset)
            #body.append(insert_node)
            #self.current_lineno=insert_node.lineno
            #self.add_num+=1
            
def get_max_lineno(body):
    stmt=body[-1]
    if isinstance(stmt, (ast.While, ast.For, ast.If)):
        if stmt.orelse:
            return get_max_lineno(stmt.orelse.body)
        else:
            return get_max_lineno(stmt.body)
    else:
        return stmt.lineno
    
class InsertReturn(object):
    def __init__(self, env):
        self.node=env.node
        
    def run(self):
        self.insert_return(self.node.body[0].body)
    
    def insert_return(self, body):
        stmt=body[-1]
        if isinstance(stmt, ast.Pass):
            return_none_node=ast.Return(value=ast.Name(id='None'))
            ast.copy_location(return_none_node, stmt)
            body[-1]=return_none_node
        elif not isinstance(stmt, ast.Return):
            assert not isinstance(stmt, (ast.For, ast.While))
            lineno=get_max_lineno(body)+1
            return_none_node=ast.Return(value=ast.Name(id='None', lineno=lineno), lineno=lineno) #fix col_offset!!
            body.append(return_none_node)

#called after cfa            
class ClearUnreachedNode(object):
    def __init__(self, env):
        self.env=env
        self.cf=env.cf
        self.node=env.node
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
            #if entry_offset==-1: #
            #    continue            
            block=self.cf.blocks[entry_offset]
            for offset in block.body:
                node=self.map_offset_node[offset]
                node.keep=True
                
    def run(self):
        func_body=self.node.body[0].body
        self.mark()
        for node in func_body[:]:
            self.visit_and_clear(node, func_body)


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
    def run(self):
        for name,type,offset,ty in self.env.load_coerce_infos:    
            #type<-ty below offset
            insert_node=ast.Assign(targets=[ast.Name(id=name, 
                                                ctx=ast.Store(), 
                                                typeinfo=TypeInfo(type))],
                                   value=ast.Call(func=ast.Name(id='<%s>'%type),
                                                  args=[ast.Name(id=name, 
                                                        ctx=ast.Load(), 
                                                        typeinfo=TypeInfo(ty))],
                                                  keywords=[],
                                                  starargs=None,
                                                  kwargs=None))
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
                
        

class RewriteName(ast.NodeVisitor):
    def __init__(self, env):
        self.node=env.node
        self.used_types=env.used_types
        self.map_name_typeinfo={}
        
    def run(self):
        self.visit(self.node.body[0])
           
    def get_order(self, name, type):
        if name in self.map_name_typeinfo:
            types, next_order=typeinfo=self.map_name_typeinfo[name]
            if type in types:
                return types[type]
            else:
                types[type]=next_order
                typeinfo[1]+=1
                return next_order
        else:
            self.map_name_typeinfo[name]=[{type:0},1]
            return 0
        
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
                    order=self.get_order(name, type)
                    node.id="%s_%s"%(name, order)
                    self.used_types.add((name, type, order)) #original type
            
class RewriteSubscript(ast.NodeTransformer):
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
                
        insert_below(-1, defination_nodes, self.env)
        #self.insert_defination(defination_nodes)
                    
    def insert_defination(self, defination_nodes):
        func_body=self.node.body[0].body
        for node in defination_nodes[::-1]:
            func_body.insert(0, node)
                    