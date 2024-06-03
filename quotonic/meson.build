# Initialize project
# Dc_args=-Og enables optimizations for the compilation
project(
  'quotonic',
  'cython',
  'c',
  default_options: ['buildtype=debugoptimized', 'optimization=g'],
  license: 'MIT',
  meson_version: '>=0.64.0',
  version: '0.0.1'
  )

# Get the Python installation
# pure: false ensures that the extensions are installed alongside pure python modules
py = import('python').find_installation(pure: false)

# Choose compiler arguments based on compiler found
# -march=native tells the compiler to tune generated code for the micro-architecture and ISA extensions of the host CPU
# -ffast-math (/fp:fast) can improve speed with no noticeable loss of reliability
c_compiler_arg_syntax = meson.get_compiler('c').get_argument_syntax()
if c_compiler_arg_syntax == 'gcc'
  if build_machine.system() == 'windows' and build_machine.cpu_family() == 'x86_64'
    c_args = ['-march=native', '-ffast-math', '-DMS_WIN64']
    message('Using gcc compiler argument syntax for mingw-win64.')
  else
    c_args = ['-march=native', '-ffast-math']
    message('Using gcc compiler argument syntax.')
  endif
elif c_compiler_arg_syntax == 'msvc'
  c_args = ['/fp:fast']
  message('Using msvc compiler argument syntax.')
else
  c_args = []
endif

# Declare the OpenMP dependency
if build_machine.system() == 'windows'
  openmp_version = '>=5.0'
else
  openmp_version = '>=4.5'
endif
openmp = dependency(
  'openmp',
  language: 'c',
  version: openmp_version,
  not_found_message: 'OpenMP could not be found for the C compiler.',
  required: false
  )

# Get NumPy include directory
incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

# Install the pure Python sources
py.install_sources(
    [
        'src/quotonic/__init__.py',
        'src/quotonic/clements.py',
        'src/quotonic/kerr.py',
        'src/quotonic/qd.py',
        'src/quotonic/qpnn.py',
        'src/quotonic/misc.py',
        'src/quotonic/utilities.py',
        'src/quotonic/training_sets.py'
    ],
    subdir: 'quotonic'
)

# Compile and install the Cython extensions
py.extension_module(
    'fock',
    'src/quotonic/fock.pyx',
    c_args: c_args,
    dependencies: [openmp],
    include_directories: [incdir_numpy],
    subdir: 'quotonic',
    install: true,
)

py.extension_module(
    'aa',
    'src/quotonic/aa.pyx',
    c_args: c_args,
    dependencies: [openmp],
    include_directories: [incdir_numpy],
    subdir: 'quotonic',
    install: true,
)