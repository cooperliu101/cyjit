import ast

class CFBlock(object):
    def __init__(self, offset):
        self.offset = offset
        self.body = []
        self.outgoing = set()
        self.incoming = set()
        self.terminating = False

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
        self.break_=False

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
            newblock.incoming.add(block.offset)
            
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
            self._curblock.body.append(stmt.lineno)
            yield stmt        

    def run(self):
        self._visit_body(self.node.body[0].body)
       
    def _visit_body(self, body):
        self._force_new_block=True
        outgoing_blocks=[]
        for stmt in self._iter_body(body):
            fname = "visit_%s" % type(stmt).__name__
            fn = getattr(self, fname, None)
            if fn is not None:
                outgoing_blocks=fn(stmt)
            else:
                outgoing_blocks=[self._curblock]         
            self._incoming_blocks=outgoing_blocks
            
            if self.break_:
                self.break_=False
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
            entry_blocks[0].incoming.add(block.offset)
        outgoing_blocks=entry_blocks
        self._loop_next_blocks.pop()
        self._force_new_block=True
        return outgoing_blocks
    
    visit_For=visit_While
    
    def jump(self, target):
        self._curblock.outgoing.add(target.offset)
        
    def visit_Break(self, node):
        self.break_=True
        self.jump(self._loop_next_blocks[-1])
        outgoing_blocks=[]
        return outgoing_blocks
    
    def visit_Return(self, node):
        self._curblock.terminating = True
        self.break_=True
        outgoing_blocks=[]
        return outgoing_blocks
      
           


s='''#1
def f(): #2
  a=1 #3
  while a>1:#4
    while a>1: #5
      return a #6
    pass #7
  return a #8
'''       
        

           
           
#s=''' #1
#def f(): #2
  #if a>1: #3
    #if a>1: #4
      #b=1 #5
  #b=3 #6
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
  #if a>1: #3
    #b=1 #4
  #else: #5
    #if a>1: #6
      #b=2 #7
    #else: #8
      #if a>1: #9
        #b=3 #10
  #b=3 #11
#'''
node=ast.parse(s)
cf=myControlFlowAnalysis(node)
cf.run()
print cf.blocks