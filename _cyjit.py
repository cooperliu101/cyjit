from inspect import getsource, getargspec
from cStringIO import StringIO
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from Cython.Build.Dependencies import cythonize
import os
import sys
import cython
import imp
import hashlib
import ast
import re
    


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

def get_body(src):
    ix = src.index(':')
    #lambda function not supported now
    #if src[:5] == 'lambda':
    #    return "return %s\n" % src[ix+1:]
    #else:
    assert src[ix+1]=='\n'
    return src[ix+2:]


    
#def parse_one_cdeclare(line):    
    #L=len(line)
    #ix=0
    #while line[ix]!=' ':
        #ix+=1
        #if ix==L:
            #raise Exception("'%s' do not have a blank"%line)
    #type_name=line[:ix]
    #vars_name=line[ix+1].strip()
    #return type_name, vars_name

#def parse_cdeclare(src):
    #for line in src.strip().split('\n'):
        #line=line.strip()
        #if line:
            #yield parse_one_cdeclare(line)

def parse_sig(sig):
    assert sig[-1]==')'
    return_type=''
    rix=sig.index('(')
    return_type=sig[:rix]
    sig=sig[rix+1:]
    L=len(sig)
    arg_type=''
    arg_types=[]
    ix=0
    while ix<L:
        if sig[ix]=='[':
            arg_type+=sig[ix]
            ix+=1           
            
            while sig[ix]!=']':
                arg_type+=sig[ix]
                ix+=1
                if ix==L:
                    if arg_type[-1]==')':
                        arg_type=arg_type[:-1]
                    raise Exception("array type of '%s' is not complete, missing ']'"%arg_type)
            arg_type+=sig[ix]
            arg_types.append(arg_type)
            arg_type=''
            ix+=1
            
        if sig[ix]==',':
            if arg_type:
                arg_types.append(arg_type)
            arg_type=''
            ix+=1       
            
        if sig[ix]==')':
            if arg_type:
                arg_types.append(arg_type)
            return return_type, arg_types

        arg_type+=sig[ix]
        ix+=1
        
#print parse_sig('int (int[:, ::1], double)')

def _get_build_extension():
    dist = Distribution()
    # Ensure the build respects distutils configuration by parsing
    # the configuration files
    config_files = dist.find_config_files()
    dist.parse_config_files(config_files)
    build_extension = build_ext(dist)
    build_extension.finalize_options()
    return build_extension

class JittedFunc(object):
    def __init__(self, c_func, py_func, func_name, module_name, module_dir, module_cython_code):        
        self.c_func=c_func
        self.py_func=py_func
        self.func_name=func_name
        self.module_name=module_name
        self.module_dir=module_dir
        self.module_cython_code=module_cython_code
        
    def __call__(self, *args, **kwds):
        return self.c_func(*args, **kwds)
    
def rebuild_attribute_func(node):
    names=[]
    def parse(node):
        names.append(node.attr)
        if isinstance(node.value, ast.Attribute):
            parse(node.value)
        elif isinstance(node.value, ast.Name):
            names.append(node.value.id)
    parse(node)
    names.reverse()
    return '.'.join(names)
  
class VisitJittedFunc(ast.NodeVisitor):
    def visit_in_context(self, node, func_args, context, func_list, local_names, load_names):
        self.func_args=func_args
        self.context=context
        self.func_list=func_list
        self.local_names=local_names
        self.load_names=load_names
        super(VisitJittedFunc, self).visit(node)
    def visit_Name(self, node):
        name=node.id
        if isinstance(node.ctx, ast.Store):
            self.local_names.add(name)
            
        if isinstance(node.ctx, ast.Load) and (name not in self.local_names) and (name not in self.func_args) and (name in self.context):
            print name, 'load from globals'
            self.load_names.add(name)
            #try:
                #value=eval(name, self.context, {})
            #except:
                #print 'can not find Name %s in globals'%name
            #else:
                #print 'find Name %s in globals'%name
                
    
    def visit_Call(self, node):
        #print ast.dump(node)
        assert isinstance(node.func, (ast.Name, ast.Attribute))
        
        if isinstance(node.func, ast.Name):
            try:
                func=eval(node.func.id, self.context, {})
            except:
                print 'can not find func %s'%node.func.id
            else:
                if isinstance(func, DeferedJittedFunc):
                    print 'jitedfunc',func.func_name
                    self.func_list.add((func, 'direct', ''))
                else:
                    self.visit_Name(node.func)
                    
        elif isinstance(node.func, ast.Attribute):
            attribute_func_name=rebuild_attribute_func(node.func)
            try:
                func=eval(attribute_func_name, self.context, {})
            except:
                print 'can not find func %s'%attribute_func_name
            else:
                if isinstance(func, DeferedJittedFunc):
                    #print 'jitedfunc',func
                    self.func_list.add((func, 'attribute', attribute_func_name))            

jvisit=VisitJittedFunc().visit_in_context

jitted_funcs=None

class DeferedJittedFunc(object):
    #__slots__=['sig', 'options', 'py_func', 'func_globals', 'func_name', 'module_name', 'module_dir', 'pyx_src', 'pxd_src']
    def __init__(self, sig, options, py_func):
        self.sig=sig
        self.options=options
        self.py_func=py_func
        self.c_func=None
        self.func_globals=py_func.func_globals
        self.func_name=py_func.__name__
        self.pyx_buf=StringIO()
        self.pxd_buf=StringIO()
        self.pyx_src=''
        self.pxd_src=''
        self.walked_jitted_funcs=[]
        self.called_jitted_funcs=set()
        self.load_names=set()
        self.local_names=set()
        file_name=self.func_globals.get('__file__')
        if file_name:
            self.module_name=os.path.splitext(os.path.basename(file_name))[0]+'__'+self.func_name
        else:
            self.module_name='interperater__'+self.func_name
        self.module_dir=''       
        self.compile_state='not_compiled'
    def putln(self, statment='', dest='pyx'):
        if dest=='pyx':
            self.pyx_buf.write(statment+"\n")
        elif dest=='pxd':
            self.pxd_buf.write(statment+'\n')
       
    
    def compile(self, walked_jitted_funcs=None): #compile and load c_func
        if self.compile_state=='not_compiled':
            #print self.func_name, 'compiling'
            self.compile_state='compiling'
            if walked_jitted_funcs==None:
                self.walked_jitted_funcs=[]
                
            py_src=get_func_source(self.py_func)
            args=get_args(self.py_func)                                                                                                                                    
            body=get_body(py_src)
            #print py_src
            indent=get_indent(body)
            func_def='cpdef'        
    
            func_ast=ast.parse(py_src)
            jvisit(func_ast, args, self.func_globals, self.called_jitted_funcs, self.local_names, self.load_names)
            
    
            #print called_jitted_funcs
            for jitted_func, mode, extra in self.called_jitted_funcs:
                
                if mode=='direct':
                    self.putln("from %s cimport %s__c"%(jitted_func.module_name, jitted_func.func_name))
                    body=re.sub(r'\b%s *\('%jitted_func.func_name, '%s__c('%(jitted_func.func_name), body)
                    
                #elif mode=='attribute':#fix
                    #self.putln('cimport %s'%(jitted_func.module_name,))
                    #attribute_func_name=extra
                    ##
                    #body=re.sub('%s *\('%attribute_func_name, '%s.%s__c('%(jitted_func.module_name, jitted_func.func_name), body)
    
            self.putln()
    
    
            #process signature
            formated_sig=self.sig.replace(' ','')
            if formated_sig:
                return_type, arg_types=parse_sig(formated_sig)
                assert len(args)==len(arg_types), "the function %s has %s args but the signature has %s args "%(func_name, len(args), len(arg_types))
                func_args=[]
                for arg_name, arg_type in zip(args, arg_types):
                    func_args.append('%s %s'%(arg_type, arg_name))
                func_args=", ".join(func_args)
            else:
                func_args=', '.join(args)
                return_type=''
                
            #directives
            #print "in"
            directive_decorates=[]
            directives=['wraparound', 'boundscheck']
            for directive in directives:
                value=self.options.get(directive)
                if value != None:
                    directive_decorates.append('@cython.%s(%s)\n'%(directive, value))
            if directive_decorates:
                self.putln('cimport cython')
                for decorate in directive_decorates:
                    self.putln(decorate)
                    
            nogil='nogil' if self.options.get("nogil") else ''
            #head
            #add like 'nogil'
            cfunc_head='cdef %s %s__c(%s) %s'%(return_type, self.func_name, func_args, nogil)
            pyfunc_head='def %s__p(%s)'%(self.func_name, ', '.join(args))
           
            #func_head_1='\n%s %s(%s)%s'%('cdef', func_name, func_args, '')
            self.putln(cfunc_head,'pxd')
            #self.pxd_buf.write(func_init_head+"\n")
            
            if self.load_names:
                init_func_head='def %s_init(f_globals)'%(self.func_name)
                globals_name='_%s_GLOBALS'%self.func_name.upper()
                self.putln('%s = None\n'%globals_name)
                self.putln(init_func_head+':')
                self.putln(indent+'global %s'%globals_name)
                self.putln(indent+'%s = f_globals'%globals_name)
                self.putln()
            
            
            self.putln(cfunc_head+':')
            #cdef vars
            locals=self.options.get('locals')
            if type(locals) is str:
                self.putln(indent+"cdef:")
                for line in locals.strip().split('\n'):
                    line=line.strip()
                    if line[-1]==';':
                        line=line[:-1]
                    self.putln(indent*2+line)
                
            #if type(locals) is str:
            #    for type_name, vars_name in parse_cdeclare(locals):
            #        cdef=indent+'cdef %s %s\n'%(type_name, vars_name)
            #        pyx_buf.write(cdef)
            
            #body
            #self.pyx_buf.write(indent+'%s=sys._getframe(1).f_locals\n'%func_globals_name)
            if self.load_names:
                self.putln(indent+'#load globals')        
                for name in self.load_names:
                    self.putln(indent+"%s = %s['%s']"%(name, globals_name, name))
                self.putln()
            
            #self.putln(indent+'print "%s", %s["__file__"]'%(self.func_name, func_globals_name))
            self.putln(body)
            self.putln()
            
            self.putln(pyfunc_head+':')
            self.putln(indent+'return %s__c(%s)'%(self.func_name, ', '.join(args)))
            
            
            self.pyx_src=self.pyx_buf.getvalue()
            self.pxd_src=self.pxd_buf.getvalue()
            
            file_name=self.func_globals.get('__file__')
            if file_name:
                func_file_dir=os.path.dirname(os.path.abspath(file_name))
            else:
                func_file_dir=os.getcwd()
                
            cython_compile_dir=os.path.join(func_file_dir, '__cython_compile__')
            if not os.path.exists(cython_compile_dir):
                os.mkdir(cython_compile_dir)
                
            key = self.pyx_src+str(sys.version_info)+sys.executable+cython.__version__
            self.module_dir=os.path.join(cython_compile_dir, self.module_name+'__'+hashlib.md5(key.encode('utf-8')).hexdigest() )        
            if not os.path.exists(self.module_dir):
                os.mkdir(self.module_dir)
    
            
            
            pyx_file=os.path.join(self.module_dir, self.module_name+'.pyx')
            pxd_file=os.path.join(self.module_dir, self.module_name+'.pxd')
            so_ext='.pyd'        
            so_file=os.path.join(self.module_dir, self.module_name+so_ext)        
            init_file=os.path.join(self.module_dir, '__init__.py')
            
            if not os.path.exists(so_file):
                fw=open(pyx_file,"w")
                fw.write(self.pyx_src)
                fw.close()
                
                fw=open(pxd_file,"w")
                fw.write(self.pxd_src)
                fw.close()
                
                fw=open(init_file, "w")
                fw.close()
                
            
            for jitted_func,_,_ in self.called_jitted_funcs:
                #print jitted_func.func_name, jitted_func.compile_state
                if jitted_func.compile_state=='not_compiled':
                    if walked_jitted_funcs==None:
                        jitted_func.compile(self.walked_jitted_funcs)
                    else:
                        jitted_func.compile(walked_jitted_funcs)
            
            
            #check
            if not os.path.exists(so_file):            
                
                extension = Extension(name = self.module_name,
                                      sources = [pyx_file],
                                      #include_dirs = c_include_dirs,
                                      #extra_compile_args = cflags
                                     )
                            
                #find called_jitted_funcs's module path
                cython_include_dirs=set()
                for jitted_func, _, _ in self.called_jitted_funcs:
                    cython_include_dirs.add(jitted_func.module_dir)
                print 'cython_include_dirs', self.func_name, cython_include_dirs
                build_extension = _get_build_extension()
                build_extension.extensions = cythonize([extension],
                                                       annotate=True,
                                                       include_path=list(cython_include_dirs), 
                                                       #quiet=quiet
                                                       )
                temp_dir=os.path.join(cython_compile_dir, "__build_temp__")
                build_extension.build_temp = temp_dir 
                build_extension.build_lib  = self.module_dir
                #print "build"
                build_extension.run()
            
            self.compile_state='compiled'
            self.so_file=so_file
            
            if walked_jitted_funcs==None:
                self.walked_jitted_funcs.append(self)
            else:
                walked_jitted_funcs.append(self)

        
    def __call__(self, *args, **kwds):
        #print self.compile_state
        if self.compile_state=='not_compiled':
            self.compile()
            assert self.compile_state=='compiled'
         
        if self.compile_state!='loaded':
            path=set()
            
            for jitted_func in self.walked_jitted_funcs:
                #print jitted_func.func_name
                path.add(jitted_func.module_dir)
            sys.path.extend(path)
            
            for jitted_func in self.walked_jitted_funcs:
                compiled_module=imp.load_dynamic(jitted_func.module_name, jitted_func.so_file)
                if jitted_func.load_names:
                    func_init=getattr(compiled_module, "%s_init"%jitted_func.func_name)
                    func_init(jitted_func.func_globals)
                
                #print 'load',jitted_func.module_name
                c_func=getattr(compiled_module, "%s__p"%jitted_func.func_name)      
                jitted_func.c_func=c_func        
                jitted_func.compile_state='loaded'                
 
            assert self.compile_state=='loaded'
        return self.c_func(*args, **kwds)
    def __str__(self):
        return "(JittedFunction %s, compile_state: %s)"%(self.func_name, self.compile_state)
    
    __repr__=__str__



def newjit(sig, **options): #move jit into call
    def wrap(py_func):
        
        return DeferedJittedFunc(sig, options, py_func)
    return wrap


def jit(sig, **kwds):
    
    def wrap(py_func):
        #clear buf
        pyx_buf.seek(0)
        pyx_buf.truncate()
        
        pxd_buf.seek(0)
        pxd_buf.truncate()  
        
        called_jitted_funcs=set()
        local_names=set() #in py_func
        load_names=set() #in py_func
        
        func_name=py_func.__name__
        func_globals_name='func_%s_globals'%func_name
        #func_globals=sys._getframe(1).f_locals #get caller's scope
        func_globals=py_func.func_globals

        py_src=get_func_source(py_func)
        args=get_args(py_func)                                                                                                                                    
        body=get_body(py_src)
        #print py_src
        indent=get_indent(body)
        func_def='cpdef'        

        func_ast=ast.parse(py_src)
        jvisit(func_ast, func_globals, called_jitted_funcs, local_names, load_names)
        #print called_jitted_funcs
        for jitted_func, mode, extra in called_jitted_funcs:
            
            if mode=='direct':
                pyx_buf.write("from %s cimport %s\n"%(jitted_func.module_name, jitted_func.func_name))
            elif mode=='attribute':
                pyx_buf.write('cimport %s\n'%(jitted_func.module_name,))
                attribute_func_name=extra
                #
                body=re.sub('%s *\('%attribute_func_name, '%s.%s('%(jitted_func.module_name, jitted_func.func_name), body)

        pyx_buf.write('\n')
        
        pyx_buf.write('''
import sys
%s=sys._getframe(1).f_locals
'''%func_globals_name)
        for name in load_names:
            pyx_buf.write("%s=%s.get('%s')\n"%(name, func_globals_name, name))
        pyx_buf.write('\n')
        
        #process signature
        
        formated_sig=sig.replace(' ','')
        if formated_sig:
            return_type, arg_types=parse_sig(formated_sig)
            assert len(args)==len(arg_types), "the function %s has %s args but the signature has %s args "%(func_name, len(args), len(arg_types))
            func_args=[]
            for arg_name, arg_type in zip(args, arg_types):
                func_args.append('%s %s'%(arg_type, arg_name))
            func_args=", ".join(func_args)
        else:
            func_args=', '.join(args)
            return_type=''
            
        #directives
        #print "in"
        directive_decorates=[]
        directives=['wraparound', 'boundscheck']
        for directive in directives:
            value=kwds.get(directive)
            if value != None:
                directive_decorates.append('@cython.%s(%s)\n'%(directive, value))
        if directive_decorates:
            pyx_buf.write('cimport cython\n\n')
            for decorate in directive_decorates:
                pyx_buf.write(decorate)
                
        nogil='nogil' if kwds.get("nogil") else ''
        #head
        #add like 'nogil'
        func_head='%s %s %s(%s) %s'%(func_def, return_type, func_name, func_args, nogil)
        #func_head_1='\n%s %s(%s)%s'%('cdef', func_name, func_args, '')
        pxd_buf.write(func_head+"\n")
        pyx_buf.write(func_head+':\n')
        #cdef vars
        locals=kwds.get('locals')
        if type(locals) is str:
            pyx_buf.write(indent+"cdef:\n")
            for line in locals.strip().split('\n'):
                line=line.strip()
                if line[-1]==';':
                    line=line[:-1]
                pyx_buf.write(indent*2+line+'\n')
            
        #if type(locals) is str:
        #    for type_name, vars_name in parse_cdeclare(locals):
        #        cdef=indent+'cdef %s %s\n'%(type_name, vars_name)
        #        pyx_buf.write(cdef)
        
        #body
        pyx_buf.write(body+'\n')
        
        #build
        pyx_src=pyx_buf.getvalue()
        pxd_src=pxd_buf.getvalue()
        
        base_name=os.path.basename(func_globals['__file__'])
        file_name=os.path.splitext(base_name)[0]
        key = pyx_src+str(sys.version_info)+sys.executable+cython.__version__
        hashed_module_name = file_name+'_'+func_name+'__'+hashlib.md5(key.encode('utf-8')).hexdigest()
        caller_file_dir=os.path.dirname(os.path.abspath(func_globals['__file__']))
        #print 'caller_file_dir', caller_file_dir
        module_dir=os.path.join(caller_file_dir, '__cython_compile__')
        if not os.path.exists(module_dir):
            os.mkdir(module_dir)
        
        pyx_file=os.path.join(module_dir, hashed_module_name+'.pyx')
        pxd_file=os.path.join(module_dir, hashed_module_name+'.pxd')
        so_ext='.pyd'        
        so_file=os.path.join(module_dir, hashed_module_name+so_ext)        
        init_file=os.path.join(module_dir, '__init__.py')
        
        #check
        if not os.path.exists(so_file):            
        
            fw=open(pyx_file,"w")
            fw.write(pyx_src)
            fw.close()
            
            fw=open(pxd_file,"w")
            fw.write(pxd_src)
            fw.close()
            
            fw=open(init_file, "w")
            fw.close()
            
            
            extension = Extension(name = hashed_module_name,
                                  sources = [pyx_file],
                                  #include_dirs = c_include_dirs,
                                  #extra_compile_args = cflags
                                 )
            
            #find called_jitted_funcs's module path
            cython_include_dirs=[]
            for jitted_func, _, _ in called_jitted_funcs:
                cython_include_dirs.append(jitted_func.module_dir)
            #print 'cython_include_dirs', cython_include_dirs
            build_extension = _get_build_extension()
            build_extension.extensions = cythonize([extension],
                                                   annotate=True,
                                                   include_path=cython_include_dirs, 
                                                   #quiet=quiet
                                                   )
            temp_dir=os.path.join(module_dir, "__build_temp__")
            build_extension.build_temp = temp_dir
            build_extension.build_lib  = module_dir
            #print "build"
            build_extension.run()
    
        compiled_module=imp.load_dynamic(hashed_module_name, so_file)        
        c_func=getattr(compiled_module, func_name)
        
        return JittedFunc(c_func, 
                          py_func, 
                          func_name, 
                          hashed_module_name, 
                          module_dir, 
                          pyx_src)
    return wrap    

#def make(module_name):
    #pyx_buf=StringIO()
    #pxd_buf=StringIO()
    
    #is_cimport_cython=[False] #ugly hack
    
    #jited_func_names=[]
    #loaded_func=[]
    
    #called_jitted_funcs=[]
    
    #def jit(sig, **kwds):
        #def wrap(func):
            #func_name=func.__name__
            #jited_func_names.append(func_name)
            
            #src=get_func_source(func)
            #args=get_args(func)                                                                                                                                    
            #body=get_body(func)
            #print src
            
            #func_ast=ast.parse(src)
            #context=sys._getframe(1).f_locals #get caller's scope
            
            #jvisit(func_ast, context, called_jitted_funcs)
            #print called_jitted_funcs
            #for jitted_func in called_jitted_funcs:
                #pyx_buf.write("from %s cimport %s\n"%(jitted_func.module_name, jitted_func.func_name))
            
            #indent=get_indent(body)
            #func_def='cpdef'
            ##process signature
            
            #formated_sig=sig.replace(' ','')
            #if formated_sig:
                #return_type, arg_types=parse_sig(formated_sig)
                #assert len(args)==len(arg_types), "the function %s has %s args but the signature has %s args "%(func_name, len(args), len(arg_types))
                #func_args=[]
                #for arg_name, arg_type in zip(args, arg_types):
                    #func_args.append('%s %s'%(arg_type, arg_name))
                #func_args=", ".join(func_args)
            #else:
                #func_args=', '.join(args)
                #return_type=''
            ##directives
            ##print "in"
            #directive_decorates=[]
            #directives=['wraparound', 'boundscheck']
            #for directive in directives:
                #value=kwds.get(directive)
                #if value != None:
                    #directive_decorates.append('@cython.%s(%s)\n'%(directive, value))
            #if directive_decorates:
                #if not is_cimport_cython[0]:
                    #pyx_buf.write('cimport cython\n\n')
                    #is_cimport_cython[0]=True
                    
                #for decorate in directive_decorates:
                    #pyx_buf.write(decorate)
            #nogil='nogil' if kwds.get("nogil") else ''
            ##head
            ##add like 'nogil'
            #func_head='%s %s %s(%s) %s'%(func_def, return_type, func_name, func_args, nogil)
            ##func_head_1='\n%s %s(%s)%s'%('cdef', func_name, func_args, '')
            #pxd_buf.write(func_head+"\n\n")
            #pyx_buf.write(func_head+':\n')
            ##cdef vars
            #locals=kwds.get('locals')
            #if type(locals) is str:
                #pyx_buf.write(indent+"cdef:\n")
                #for line in locals.strip().split('\n'):
                    #line=line.strip()
                    #if line[-1]==';':
                        #line=line[:-1]
                    #pyx_buf.write(indent*2+line+'\n')
                
            ##if type(locals) is str:
            ##    for type_name, vars_name in parse_cdeclare(locals):
            ##        cdef=indent+'cdef %s %s\n'%(type_name, vars_name)
            ##        pyx_buf.write(cdef)
            ##body
            #pyx_buf.write(body+'\n')
            ##print buf.getvalue()
            #return func
        #return wrap
    
    #def build():
        
        #caller_frame=sys._getframe(1)
        #caller_locals=caller_frame.f_locals
        
        #pyx_src=pyx_buf.getvalue()
        #pxd_src=pxd_buf.getvalue()
        
        #key = pyx_src+str(sys.version_info)+sys.executable+cython.__version__
        #hashed_module_name = module_name+'__'+hashlib.md5(key.encode('utf-8')).hexdigest()
        #caller_file_dir=os.path.dirname(os.path.abspath(caller_locals['__file__']))
        ##print 'caller_file_dir', caller_file_dir
        #module_dir=os.path.join(caller_file_dir, '__cython_compile__')
        #if not os.path.exists(module_dir):
            #os.mkdir(module_dir)
        
        #pyx_file=os.path.join(module_dir, hashed_module_name+'.pyx')
        #pxd_file=os.path.join(module_dir, hashed_module_name+'.pxd')
        #so_ext='.pyd'        
        #so_file=os.path.join(module_dir, hashed_module_name+so_ext)        
        #init_file=os.path.join(module_dir, '__init__.py')
        
        #if not os.path.exists(so_file):            
        
            #fw=open(pyx_file,"w")
            #fw.write(pyx_src)
            #fw.close()
            
            #fw=open(pxd_file,"w")
            #fw.write(pxd_src)
            #fw.close()
            
            #fw=open(init_file, "w")
            #fw.close()
            
            
            #extension = Extension(name = hashed_module_name,
                                  #sources = [pyx_file],
                                  ##include_dirs = c_include_dirs,
                                  ##extra_compile_args = cflags
                                 #)
            
            
            #cython_include_dirs=[]
            #for func in called_jitted_funcs:
                #cython_include_dirs.append(func.module_dir)
            ##print 'cython_include_dirs', cython_include_dirs
            #build_extension = _get_build_extension()
            #build_extension.extensions = cythonize([extension],
                                                   #annotate=True,
                                                   #include_path=cython_include_dirs, 
                                                   ##quiet=quiet
                                                   #)
            #temp_dir=os.path.join(module_dir, "__build_temp__")
            #build_extension.build_temp = temp_dir
            #build_extension.build_lib  = module_dir
            ##print "build"
            #build_extension.run()
        ##print "load", hashed_module_name, so_file
        #compiled_module=imp.load_dynamic(hashed_module_name, so_file)
        
        #while jited_func_names:
            #func_name=jited_func_names.pop()
            #caller_locals[func_name]=JittedFunc(getattr(compiled_module, func_name), 
                                                   #caller_locals[func_name],
                                                   #func_name,
                                                   #hashed_module_name,
                                                   #module_dir,
                                                   #pyx_src,
                                                   #)
    
    #def load(*args, **kwds):    
        #for func in args:
            #assert isinstance(func, JittedFunc)
            #loaded_func.append(func)
            #pyx_buf.write("from %s cimport %s\n"%(func.module_name, func.func_name))
    
    #def inline(statment):
        #pyx_buf.write(statment+'\n')
    
    #return inline, load, jit, build

if __name__=='__main__':
    @jit('',locals='int i')
    def f():
        for i in range(10):
            print i
  