from inspect import getsource, getargspec
from cStringIO import StringIO
from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext
from Cython.Build.Dependencies import strip_string_literals, cythonize, cached_function
import os
import sys
import cython
import imp
import hashlib

    


def get_args(func):
    return getargspec(func).args

def get_body(func):
    src_lines=[]
    keep=False
    for line in getsource(func).split('\n'):
        if line.startswith('def'):
            keep=True
        if keep and line.strip():
            src_lines.append(line.rstrip())
    src='\n'.join(src_lines)
    
    ix = src.index(':')
    if src[:5] == 'lambda':
        return "return %s\n" % src[ix+1:]
    else:
        assert src[ix+1]=='\n'
        return src[ix+2:]

def get_indent(body):      
    assert body[0] in ' \t'
    ix=1
    while body[ix] in ' \t':
        ix+=1
    return body[:ix]
    
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

class jitted_func(object):
    def __init__(self, c_func, py_func, func_name, module_name, module_dir, module_cython_code):        
        self.c_func=c_func
        self.py_func=py_func
        self.func_name=func_name
        self.module_name=module_name
        self.module_dir=module_dir
        self.module_cython_code=module_cython_code
        
    def __call__(self, *args, **kwds):
        return self.c_func(*args, **kwds)
        
def make(module_name):
    pyx_buf=StringIO()
    pxd_buf=StringIO()
    
    is_cimport_cython=[False] #ugly hack
    
    jited_func_names=[]
    loaded_func=[]
    
    def jit(sig, **kwds):
        def wrap(func):
            func_name=func.__name__
            jited_func_names.append(func_name)
            
            args=get_args(func)                                                                                                                                    
            body=get_body(func)
            indent=get_indent(body)
            func_def='cpdef'
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
                if not is_cimport_cython[0]:
                    pyx_buf.write('cimport cython\n\n')
                    is_cimport_cython[0]=True
                    
                for decorate in directive_decorates:
                    pyx_buf.write(decorate)
            nogil='nogil' if kwds.get("nogil") else ''
            #head
            #add like 'nogil'
            func_head='%s %s %s(%s) %s'%(func_def, return_type, func_name, func_args, nogil)
            #func_head_1='\n%s %s(%s)%s'%('cdef', func_name, func_args, '')
            pxd_buf.write(func_head)
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
            #print buf.getvalue()
            return func
        return wrap
    
    def build():
        
        caller_frame=sys._getframe(1)
        caller_locals=caller_frame.f_locals
        
        pyx_src=pyx_buf.getvalue()
        pxd_src=pxd_buf.getvalue()
        
        key = pyx_src+str(sys.version_info)+sys.executable+cython.__version__
        hashed_module_name = module_name+'__'+hashlib.md5(key.encode('utf-8')).hexdigest()
        caller_file_dir=os.path.dirname(os.path.abspath(caller_locals['__file__']))
        #print 'caller_file_dir', caller_file_dir
        module_dir=os.path.join(caller_file_dir, '__cython_compile__')
        if not os.path.exists(module_dir):
            os.mkdir(module_dir)
        
        pyx_file=os.path.join(module_dir, hashed_module_name+'.pyx')
        pxd_file=os.path.join(module_dir, hashed_module_name+'.pxd')
        so_ext='.pyd'        
        so_file=os.path.join(module_dir, hashed_module_name+so_ext)        
        init_file=os.path.join(module_dir, '__init__.py')
        
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
            
            
            cython_include_dirs=[]
            for func in loaded_func:
                cython_include_dirs.append(func.module_dir)
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
        #print "load", hashed_module_name, so_file
        compiled_module=imp.load_dynamic(hashed_module_name, so_file)
        
        while jited_func_names:
            func_name=jited_func_names.pop()
            caller_locals[func_name]=jitted_func(getattr(compiled_module, func_name), 
                                                   caller_locals[func_name],
                                                   func_name,
                                                   hashed_module_name,
                                                   module_dir,
                                                   pyx_src,
                                                   )
    
    def load(*args, **kwds):    
        for func in args:
            assert isinstance(func, jitted_func)
            loaded_func.append(func)
            pyx_buf.write("from %s cimport %s\n"%(func.module_name, func.func_name))
    
    def inline(statment):
        pyx_buf.write(statment+'\n')
    
    return inline, load, jit, build

if __name__=='__main__':
    pass
    