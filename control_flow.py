import ast
from copy import deepcopy as copy
import re

from numba_types import Array, int32, float64, pyobject, Tuple, is_int_tuple, none, string, i1, integer_domain, real_domain

    
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
        
from passes import InsertReturn, ClearUnreachedNode, CoerceReturn, InsertArrayInfo, InsertCoerceNode, InsertDefination, RewriteName, RewriteSubscript
from type_inference import TypeInfer

class CFBlock(object):
    def __init__(self, offset):
        self.offset = offset
        self.body = []
        self.outgoing = []
        self.incoming = []
        self.terminating = False
        self.is_loop=False
        self.context={}
        self.incoming_context={}
        #self.insert_num=0

    def __repr__(self):
        args = self.body, self.outgoing, self.incoming, self.context
        return "block(body: %s, outgoing: %s, incoming: %s, context=%s)" % args

class ControlFlowAnalysis(object):
    def __init__(self, env):
        env.cf=self
        self.env=env
        self.node=env.node
        self.map_offset_body=env.map_offset_body
        self.map_offset_node=env.map_offset_node
        self.blocks={}
        self._incoming_blocks=[] #for next block
        self._current_block=None
        self._force_new_block=True
        self._loop_test_blocks=[]
        self._loop_next_blocks=[]
        
        self._break_infos=[]
        self.walked=set()
        self.path=[]
        self.terminate_blocks=[]
        
    def _use_new_block(self, stmt):
        if type(stmt).__name__ in ['While', 'For',]: #'If', 'Break', 'Return']:
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
        self._current_block = newblock
        #self.blockseq.append(offset)

           
    def _iter_body(self, body):
        for idx,stmt in enumerate(body):
            
            #if type(stmt).__name__ in ['While', 'For']:
                #loop_next_stmt=body[idx+1]
                #offset=loop_next_stmt.lineno
                #loop_next_block=CFBlock(offset)
                #self.blocks[offset]=loop_next_block
                #self._loop_next_blocks.append(loop_next_block)
            
            if self._use_new_block(stmt):
                self._start_new_block(stmt)
                
            offset=stmt.lineno
            self._current_block.body.append(offset)
            

            self.map_offset_node[offset]=stmt
            self.map_offset_body[offset]=body
            yield stmt        

    def fill_incoming_outgoing(self):
        for offset1, block in self.blocks.iteritems():
            for offset2 in block.incoming:
                outgoing=self.blocks[offset2].outgoing
                if offset1 not in outgoing:
                    outgoing.append(offset1)
                    
            for offset2 in block.outgoing:
                incoming=self.blocks[offset2].incoming
                if offset1 not in incoming:
                    incoming.append(offset1)     
                    
    def next_offset(self, offset):
        seq=sorted(self.blocks.keys())
        next_index=seq.index(offset)+1
        assert next_index < len(seq)
        return seq[next_index]
    
    def run(self):
        funcdef_node=self.node.body[0]
        self.visit_FunctionDef(funcdef_node)
        self.fill_incoming_outgoing()
        
        #handle break
        for break_block, loop_test_block in self._break_infos:
            for offset in loop_test_block.outgoing:
                if offset != self.next_offset(loop_test_block.offset):
                    break_block.outgoing.append(offset)
                    self.blocks[offset].incoming.append(break_block.offset)
                
        #walk through the blocks
        self.walk()
        #find terminated_block
        for offset, block in self.blocks.items():
            if not block.outgoing:
                self.terminate_blocks.append(offset)
                
        
                    
    def walk(self):
        start=min(self.blocks.keys())
        for bo in self.blocks[start].outgoing:
            self._walk(start, bo)
    
    def _walk(self, b1, b2):
        if (b1,b2) in self.walked:
            return
        self.walked.add((b1,b2))
        self.path.append((b1,b2))
        for bo in self.blocks[b2].outgoing:
            self._walk(b2, bo)
                    
       
    def _visit_body(self, body):
        self._force_new_block=True
        exit_blocks=[]
        for stmt in self._iter_body(body):
            stmt_type=type(stmt).__name__
            fname = "visit_%s" % stmt_type
            fn = getattr(self, fname, None)
            if fn is not None:
                exit_blocks=fn(stmt)
            else:
                exit_blocks=[self._current_block]         
            self._incoming_blocks=exit_blocks #incoming_blocks is for next stmt
            
            if stmt_type in ['Break', 'Return', 'Continue']:
                return exit_blocks   
            
        return exit_blocks
    
    def visit_FunctionDef(self, node):
        if self._use_new_block(node):
            self._start_new_block(node)
        offset=node.lineno
        self._current_block.body.append(offset)
        self.map_offset_node[offset]=node
        self.map_offset_body[offset]=None   
        self.map_offset_body[-1]=node.body #change
        self._visit_body(node.body)
    
    def visit_If(self, node):
        test_block=self._current_block
        if_body_exit_block=self._visit_body(node.body)

        if node.orelse:
            self._incoming_blocks=[test_block]
            orelse_body_exit_block=self._visit_body(node.orelse)
            exit_blocks=if_body_exit_block+orelse_body_exit_block
        else:
            exit_blocks=[test_block]+if_body_exit_block

        self._force_new_block=True
        return exit_blocks
    
    def visit_While(self, node):        
        test_block=self._current_block
        test_block.is_loop=True
        self._loop_test_blocks.append(test_block)
        
        while_body_exit_blocks=self._visit_body(node.body)
        for block in while_body_exit_blocks:
            test_block.incoming.append(block.offset)
        if node.orelse:
            self._incoming_blocks=[test_block]
            orelse_body_exit_block=self._visit_body(node.orelse)   
            exit_blocks=orelse_body_exit_block
        else:
            exit_blocks=[test_block]
            
        #self._loop_next_blocks.pop()
        self._loop_test_blocks.pop()
        self._force_new_block=True
        return exit_blocks
    
    visit_For=visit_While
    
    def jump(self, target):
        self._current_block.outgoing.append(target.offset)
        
    def visit_Break(self, node):
        #self.jump(self._loop_next_blocks[-1])
        exit_blocks=[]
        self._break_infos.append((self._current_block, self._loop_test_blocks[-1]))
        return exit_blocks
    
    def visit_Return(self, node):
        self._current_block.terminating = True
        exit_blocks=[]
        return exit_blocks
    
    def visit_Continue(self, node):
        self.jump(self._loop_test_blocks[-1])
        exit_blocks=[]
        return exit_blocks    
    


#class WalkControlFlowGraph(object):
    #def __init__(self, env):
        #self.blocks=env.cf.blocks
        #self.walked=set()
        #self.incoming={}
        #self.path=[]
        ##copy incoming
        #for offset,block in self.blocks.items():
            #self.incoming[offset]=list(block.incoming)
            
    #def _walk(self, b1, b2, r, c): #b1, b2 : offset
        #if (b1, b2) in self.walked:
            #return            
        #if r:
            #self.walked.add((b1, b2))    
        #if b2==c:
            #return 
        #self.path.append((b1, b2))
        #bk2=self.blocks[b2]
        #self.incoming[b2].remove(b1)    
        #if bk2.is_loop:
            #c=b2
        #else:
            #c=-1
        #if self.incoming[b2]: #
            #self._walk(b2, b2+1, False, c)
        #else: #all incoming block is income
            #for bo in bk2.outgoing:
                #self._walk(b2, bo, False, c)
    
    #def walk(self):
        #self._walk(1, 2, True, -1) #function_def(id:1) -> body first block(id:2) 
        #return self.path
        

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
        if keep:
            line=line[indent_size:].rstrip()
            src_lines.append(line)
    src='\n'.join(src_lines)
    return src

def compile(func, argtypes):
    
    from astunparse import Unparser
    from cStringIO import StringIO
    
    env=Env()
    source=get_func_source(func)
    argnames=get_args(func)
    print argnames
    print source
    args={arg_name:arg_type for arg_name,arg_type in zip(argnames, argtypes)}
    
    node=ast.parse(source)
    env.node=node
    
    #InsertPass(env).run()
    #InsertReturn(env).run()
    ControlFlowAnalysis(env).run()
    ClearUnreachedNode(env).run()
    print env.cf.path
    
    TypeInfer(env, args, func.func_globals).run()
    print env.cf.blocks
    InsertCoerceNode(env).run()
    #CoerceReturn(env).run()
    print ast.dump(node)
    RewriteName(env).run()
    RewriteSubscript(env).run()
    InsertArrayInfo(env).run()
    InsertDefination(env).run()

    

    buf=StringIO()
    Unparser(node,buf)
    print buf.getvalue()      
    #print get_return_type(env.return_node_infos)    
    
def cyjit(argtypes=[], restype=None):
    def wrap(func):
        compile(func, argtypes)
        
    return wrap            

if __name__ == '__main__':
    #fix me handle restype!!
    a=1
    @cyjit(argtypes=[Array(int32,2,'C')],
           restype=None,) #wather to support locals={'a':int32} like type defination? For now, No
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
    def f(b):
        s=0.0
        for i in range(10):
            for j in range(10):
                if j>2:
                    s=1 #change nested "for or while" typeinfer behavior 
                else:
                    s=j+s
                    break   
        a=s+b
        if a:
            pass
        